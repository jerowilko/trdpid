import numpy as np
import matplotlib.pyplot as plt

def plot_conv_units(units):
    num = units.shape[-1]
    sqrt_num = int(np.ceil(np.sqrt(num)))
    for i in range(num):
        plt.subplot(sqrt_num, sqrt_num, i+2)
        plt.imshow(units[:,:,0,i])
    plt.show()