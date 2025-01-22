import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# sim engine
from fungi import Well, Field
# typing
from collections.abc import Iterable
from numpy.typing import ArrayLike

# 96 well dimensions
WELL_DIAM = 6.94e3  # µm
WELL_AREA = np.pi * (WELL_DIAM/2)**2  # µm2
# Fungi.AI photo dimensions
PHOTO_DIMS = (2048, 1536)  # pixel
PIXEL_PER_SQMICRON = 32  # Fungi.AI picture scale
PHOTO_AREA = np.prod(PHOTO_DIMS) / PIXEL_PER_SQMICRON  # µm2


class SimulationRunner:
    def __init__(
            self, well_size, spore_n, timesteps, field_origin, field_size,
            out_dir="output"
    ):
        """
        Initialize the simulation runner.

        Parameters:
        -----------
        well_size : tuple[int, int]
            Dimensions of the well in pixels.
        spore_n : int
            Number of spores in the simulation.
        timesteps : int
            Total number of simulation steps.
        field_origin : tuple[int, int]
            Origin of the field in the simulation.
        field_size : tuple[int, int]
            Size of the field in pixels.
        out_dir : str
            Path to the directory where outputs will be saved.
        """
        self.well_size = well_size
        self.spore_n = spore_n
        self.timesteps = timesteps
        self.field_origin = field_origin
        self.field_size = field_size
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.start = datetime.now()
   
        # Initialize field and well
        self.field = Field(field_origin, field_size)
        self.well = Well(well_size, spore_n=spore_n, field=self.field)

        # Data for results
        self.timesteps_data = []
        self.area_data = []

    def run(self, growth_rate=2, make_gif=False, gif_name="sim.gif", log=True):
        """
        Run the simulation.

        Parameters:
        -----------
        growth_rate : int
            The growth rate of spores per timestep.
        make_gif : bool
            Whether to generate an animated GIF of the simulation.
        gif_name : str
            Name of the GIF file if `make_gif` is True.
        log : bool
            Whether to log simulation metadata.
        """
        # Initialize for GIF creation
        if make_gif:
            fig, ax = plt.subplots()
            im = self.well.render(ax)
            title = ax.text(
                    0.95, 0.95,
                    "t=0\n{:.2f} %".format(self.well.percent_area),
                    transform=ax.transAxes,
                    ha='right', va='top',
                    bbox=dict(facecolor='w', alpha=0.7),
                )
            images = [[im, title]]

        if isinstance(growth_rate, int):
            growth_rate = [growth_rate] * self.timesteps  # uniform growth

        elif isinstance(growth_rate, Iterable):
            growth_rate = list(growth_rate)

            if len(growth_rate) != self.timesteps:
                raise ValueError(
                   f'growth_rate length ({len(growth_rate)}) '
                   f'must match timesteps ({self.timesteps})'
                )

        # Run simulation timesteps
        for t, rate in enumerate(growth_rate):
            self.well.grow(rate)
            self.timesteps_data.append(t)
            self.area_data.append(self.well.percent_area)

            # Add frames for GIF if needed
            if make_gif:
                ax.axis('off')
                im = self.well.render(ax)
                title = ax.text(
                    0.95, 0.95,
                    f"t={t}\n{self.well.percent_area:.2f} %",
                    transform=ax.transAxes,
                    ha='right', va='top',
                    bbox=dict(facecolor='w', alpha=0.7),
                )
                images.append([im, title])

        # Save GIF if requested
        if make_gif:
            ani = animation.ArtistAnimation(
                fig, images, interval=100, blit=True, repeat_delay=1000
            )
            ani.save(self.out_dir / gif_name)

        # Log metadata
        if log:
            self.end = datetime.now()
            self._log_metadata()

    def _log_metadata(self):
        """
        Save simulation metadata to a JSON file.
        """
        metadata = {
            "start_date": self.start.isoformat(),
            "end_date": self.end.isoformat(),
            "well_size": self.well_size,
            "spore_n": self.spore_n,
            "field_origin": self.field_origin,
            "field_size": self.field_size,
            "timesteps": self.timesteps,
        }
        with open(self.out_dir / "log.json", "w") as f:
            json.dump(metadata, f)

    def save_results(self, filename="results.csv"):
        """
        Save simulation results to a CSV file.
        """
        df = pd.DataFrame({
            "timestep": self.timesteps_data,
            "percent_area": self.area_data,
        })
        df.to_csv(self.out_dir / filename, index=False)

    def plot_results(self, out_file="plot.png"):
        """
        Plot the percent area covered over timesteps and save the plot.
        """
        plt.figure()
        plt.plot(self.timesteps_data, self.area_data, label="Growth")
        plt.xlabel("Timesteps")
        plt.ylabel("Percent Area Covered")
        plt.title("Fungal Growth Over Time")
        plt.legend()
        plt.savefig(self.out_dir / out_file, dpi=300, bbox_inches="tight")
        plt.close()


# Utility functions for spore adjustments
def adjust_spores(well_size: tuple[int, int], spore_concentration: float):
    real_well = WELL_AREA * PIXEL_PER_SQMICRON
    sim_well = np.prod(well_size)
    spores_in_exp = spore_concentration * 1.2  # [cell/mL] * 1200 µL in well
    return int(np.ceil(sim_well / real_well * spores_in_exp))


def adjust_field_size(well_size: tuple[int, int]):
    ratio = WELL_AREA / PHOTO_AREA
    sim_area = np.prod(well_size)
    sim_field = sim_area / ratio
    field_side = int(np.ceil(np.sqrt(sim_field)))
    return (field_side, field_side)


def logistic_rates(beta: list[float], x: ArrayLike) -> ArrayLike:
    I0, L, m, x0 = beta
    numerator = np.exp(-m * (x - x0)) * m
    denominator = np.square(1 + np.exp(-m*(x - x0)))
    return numerator * L / denominator


# Example Usage
if __name__ == "__main__":
    WELL_SIZE = (2828, 2828)
    spore_concentration = 2.2e6
    runner = SimulationRunner(
        well_size=WELL_SIZE,
        spore_n=adjust_spores(WELL_SIZE, spore_concentration),
        timesteps=150,
        field_origin=(100, 100),
        field_size=adjust_field_size(WELL_SIZE),
        out_dir="logistic_rate"
    )
    gr = [1] * 50 + [3] * 80 + [1] * 20
    runner.run(growth_rate=gr, make_gif=True, gif_name="growth.gif", log=True)
    runner.save_results()
    runner.plot_results()
