#!python

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic(beta: list[float], x: ArrayLike) -> ArrayLike:
    """
    I = I0 + L / 1 + exp(-m(x-x0));

    where I0, L, m, x0 = beta
    """
    I0, L, m, x0 = beta
    return I0 + L / (1 + np.exp(-m*(x-x0)))


def model(x, a, b, c, d):
    return logistic((a, b, c, d), x)


df = pd.read_csv('spore_conc/big_sims/data.csv')
df.i = df.i.astype('category')

ind = np.arange(0, 150, 10)
f, ax = plt.subplots()
for name, subset in df.groupby('i'):
    run = subset.groupby(
        'timelapse'
    ).area.agg(['mean', 'std']).reset_index()
    errorbar = ax.errorbar(
        run.iloc[ind].timelapse,
        run.iloc[ind]['mean'],
        yerr=run.iloc[ind]['std'],
        marker='s',
        capsize=5,
        label=name
    )
    beta, _ = curve_fit(
        model,
        subset.timelapse,
        subset.area,
        [0, subset.area.max(), 1, 50],
        full_output=False,
    )
    ax.plot(
        run.iloc[ind].timelapse,
        logistic(beta, run.iloc[ind].timelapse),
        marker='',
        linestyle='--',
        color=errorbar.lines[0].get_color(),
        label="m={:.2f}; x0={:.2f}".format(beta[2], beta[3])
    )
    ax.legend()
f.savefig('bars.png', dpi=300, bbox_inches='tight')