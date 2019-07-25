import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_conv_units(units, vmin=-1, vmax=1):
    num = units.shape[-1]
    if num == 1:
        plt.imshow(units[:, :, 0, 0], vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        sqrt_num = int(np.ceil(np.sqrt(num)))
        for i in range(num):
            plt.subplot(sqrt_num, sqrt_num, i+2)
            plt.imshow(units[:,:,0,i], vmin=vmin, vmax=vmax)
            plt.colorbar()
    plt.show()

def rotate_units(u1, u2, num_steps=100, vmin=-0.15, vmax=0.15):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    im1 = ax1.imshow(u1, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(u2, vmin=vmin, vmax=vmax)
    thetas = np.linspace(0, 2*np.pi, num_steps)
    def init():
        return im1, im2
    def animate(i):
        U1 = np.cos(thetas[i])*u1 + np.sin(thetas[i])*u2
        U2 = np.sin(thetas[i]) * u1 - np.cos(thetas[i]) * u2
        im1.set_array(U1)
        im2.set_array(U2)
        return im1, im2
    ani = FuncAnimation(fig, animate, frames=num_steps,
                        init_func=init, blit=True, repeat=True, interval=100)
    plt.show()