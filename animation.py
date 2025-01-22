import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

f, ax = plt.subplots()
data = np.random.randint(0, 255, (100, 100))
im = ax.imshow(data)
title = ax.text(0.9, 0.9, '', transform=ax.transAxes)

images = [[im, title]]
for i in range(30):
    data = np.random.randint(0, 255, (100, 100))
    im = ax.imshow(data)
    title = ax.text(0.9, 0.9, str(i), transform=ax.transAxes)
    images.append([im, title])



ani = animation.ArtistAnimation(f, images, interval=50, blit=True, repeat_delay=1000)
ani.save('movie.gif')