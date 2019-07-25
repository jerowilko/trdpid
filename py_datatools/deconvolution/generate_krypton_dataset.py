import numpy as np
import matplotlib.pyplot as plt
import os
import py_datatools.deconvolution.deconvolute as deconvolute

num_evts = 8000
evt = np.zeros((540, 16, 144, 30))

bins = np.linspace(-1, 5001, 5002)
bin_centres = bins[1:] - bins[0]
vals = np.zeros(bins.shape[0] - 1)

N = 0

show_plots = True

all_units = []

i = 0

while True:
    print(i)

    try:
        deconvolute.load_krypton_event(os.path.dirname(__file__) + '/krypton_events/%d.txt' % (i + 1), evt, overwrite=True)
    except:
        break

    evt[evt!=0] -= 10.5

    v, _ = np.histogram(np.sum(evt, axis=(1, 2, 3)), bins=bins)
    vals += v

    if i == 0 or True:
        std = np.std(np.sum(evt, axis=(0, 1)), axis=1)
        time_sum = np.sum(evt, axis=(0, 1, 3))
        normed_windowed_sum_std = np.std(deconvolute.windowed_sum(np.sum(evt, axis=(0, 1)), (1, 10)), axis=1) / time_sum
        comb_statistic = std * time_sum * normed_windowed_sum_std

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

        units = [evt[pos[0], pos[1], pos[2]-2:pos[2]+3, :] for pos in positions]

        if show_plots:
            for unit in units:
                plt.imshow(unit)
                plt.show()

        all_units += units

        N += len(units)

        if i == 5:
            break

    i += 1

print('Found %d krypton interactions.' % N)


# np.save(os.path.dirname(__file__) + '/impulse_response.npy', final_unit)