import numpy as np
import matplotlib.pyplot as plt
import os
import py_datatools.deconvolution.deconvolute as deconvolute

num_evts = 30
num_events_per_plot = 5
num_skip = 8
nbins = 30
adc_spectrum = np.zeros(nbins)
evt = np.zeros((540, 16, 144, 30))

num_plotted = 0

for i in range(10000):
	deconvolute.load_krypton_event(os.path.dirname(__file__) + '/krypton_events/%d.txt' % (i + 1), evt)

	plt.subplot(1, num_events_per_plot, (i - num_skip) % num_events_per_plot + 1)
	# plt.imshow(np.sum(evt, axis=(0,1)))
	plt.imshow(evt[0,0])
	plt.clim(0, 1024)
	plt.colorbar()

	num_plotted += 1

	if num_plotted % num_events_per_plot == 0:
		plt.show()

	if num_plotted == num_evts:
		break