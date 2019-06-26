import numpy as np
import matplotlib.pyplot as plt

def plot_track(track, row=1, num_rows=1, show=True):
    for i in range(6):
        plt.subplot(num_rows, 6, (row-1)*6 + i+1)
        plt.imshow(track[i][:,::-1])
    if show:
        plt.show()