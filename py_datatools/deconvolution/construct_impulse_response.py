import numpy as np
import matplotlib.pyplot as plt
import os
import py_datatools.deconvolution.deconvolute as deconvolute

num_evts = 8000
evt = np.zeros((540, 16, 144, 30))

bins = np.linspace(-1, 5001, 5002)
bin_centres = bins[1:] - bins[0]
vals = np.zeros(bins.shape[0] - 1)


final_unit_shape = (5, 5, 12)

final_unit = np.zeros(final_unit_shape)
N = 0

show_plots = False

for i in range(num_evts):
    print(i)

    deconvolute.load_krypton_event(os.path.dirname(__file__) + '/krypton_events/%d.txt' % (i + 1), evt, overwrite=True)
    evt[evt!=0] -= 10.5

    v, _ = np.histogram(np.sum(evt, axis=(1, 2, 3)), bins=bins)
    vals += v

    if i == 0 or True:
        if show_plots:
            ax1 = plt.subplot(5, 1, 1)
            plt.imshow(np.sum(evt, axis=(0, 1)).T)
            plt.colorbar()

        std = np.std(np.sum(evt, axis=(0, 1)), axis=1)
        time_sum = np.sum(evt, axis=(0, 1, 3))
        normed_windowed_sum_std = np.std(deconvolute.windowed_sum(np.sum(evt, axis=(0, 1)), (1, 10)), axis=1) / time_sum
        comb_statistic = std * time_sum * normed_windowed_sum_std

        if show_plots:
            ax2 = plt.subplot(5, 1, 2, sharex=ax1)
            plt.plot(std)

            plt.subplot(5, 1, 3, sharex=ax2)
            plt.plot(time_sum)

            plt.subplot(5, 1, 4, sharex=ax2)
            plt.plot(normed_windowed_sum_std)

            plt.subplot(5, 1, 5, sharex=ax2)
            plt.plot(comb_statistic)

            plt.axhline(10000, c='r')
            plt.axhline(100000, c='r')
            plt.yscale('log')

        rows = np.where(np.logical_and(10000 < comb_statistic, comb_statistic < 100000))[0]

        rows = list(filter(lambda x: x < 140, rows))

        nrows = []
        j = 0
        while j < len(rows):
            k = j + 1
            while k < len(rows) and rows[k] == rows[k-1] + 1:
                k+=1

            nrows.append(rows[j] + np.nanargmax(comb_statistic[rows[j:k]]))

            j = k
        rows = nrows

        half_positions = [np.unravel_index(np.argmax(evt[:,:,row,:]), (540,16,30)) for row in rows]
        positions = [half_positions[ind][:2] + (rows[ind],) + half_positions[ind][2:] for ind in range(len(rows))]

        ### MAKE BETTER POSITION ESTIMATES BASED OFF WEIGHTED SUM.

        positions = list(filter(lambda pos: 30 - pos[3] >= 9 and pos[3] >= 3, positions))
        positions = list(filter(lambda pos: 144 - pos[2] >= 3 and pos[2] >= 3, positions))
        positions = list(filter(lambda pos: 16 - pos[1] >= 3 and pos[1] >= 3, positions))

        units = [evt[pos[0], pos[1]-2:pos[1]+3:, pos[2]-2:pos[2]+3, pos[3]-3:pos[3]+9] for pos in positions]

        if show_plots:
            plt.show()

        for unit in units:
            if show_plots:
                plt.imshow(unit[1])
                plt.show()

            final_unit = (final_unit * (N/(N+1))) + (unit / (N+1))
            N += 1

print('Average unit accross %d interactions' % N)

plt.imshow(final_unit[0])
plt.colorbar()
plt.show()

plt.imshow(final_unit[1])
plt.colorbar()
plt.show()

plt.imshow(final_unit[2])
plt.colorbar()
plt.show()

plt.imshow(final_unit[3])
plt.colorbar()
plt.show()

plt.imshow(final_unit[4])
plt.colorbar()
plt.show()

np.save(os.path.dirname(__file__) + '/impulse_response.npy', final_unit)