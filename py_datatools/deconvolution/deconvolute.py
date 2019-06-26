import numpy as np
import scipy.fftpack as fftpack
import scipy.signal
from py_datatools.datatools import *
import matplotlib.pyplot as plt
from py_datatools.plotting_tools import *

# TODO, add fftpack zero-padding optimizations. see scipy.fftpack.next_fast_len.
def deconvolve(tracks, impulse_response):
	tracklet_shape = tracks.shape[2:]
	conv_inverse_spectrum = 1 / fftpack.fftn(impulse_response, tracklet_shape)
	conv_inverse_spectrum = np.expand_dims(np.expand_dims(conv_inverse_spectrum, axis=0), axis=0)
	track_spectrum = fftpack.fftn(tracks, axes=list(range(tracks.ndim))[2:])
	return fftpack.ifftn(track_spectrum * conv_inverse_spectrum, axes=list(range(tracks.ndim))[2:]).real

def convolve(arr, filter, mode='same', *args, **kwargs):
	return scipy.signal.fftconvolve(arr, filter, mode=mode, *args, **kwargs)

def windowed_sum(arr, shape, *args, **kwargs):
	filter = np.ones(shape)
	return convolve(arr, filter, *args, **kwargs)

def resample_unit_time(impulse_response, to_time_size):
	return scipy.signal.resample(impulse_response, to_time_size, axis=-1)

def load_krypton_event(path, out=np.zeros((540, 16, 144, 30)), overwrite=True):
	data = np.genfromtxt(path, delimiter=',')

	if overwrite:
		out[::] = 0

	if len(data) == 0:
		return out

	dets = data[:, 0].astype(int)
	rows = data[:, 1].astype(int)
	cols = data[:, 2].astype(int)

	out[dets, rows, cols] += data[:, 3:]

	# out[:,:,]

	return out

if __name__=='__main__':
	dataset = '6_tracklets_large_calib_train'
	new_dataset = '6_tracklets_large_calib_deconvoluted_train'
	num_plots = 10
	baseline_value = 10.5
	offset = (2, 3)		#Since the impulse responce is centered

	tracks, info_sets = load_whole_named_dataset(dataset)
	# tracks = tracks[:10]
	impulse_response = np.load('py_datatools/deconvolution/impulse_response.npy')[2]
	imp = np.zeros((impulse_response.shape[0], 30))
	imp[:impulse_response.shape[0], :impulse_response.shape[1]] = impulse_response
	imp = resample_unit_time(imp, 24)[:,:10]

	plt.subplot(1, 2, 1)
	plt.imshow(imp)
	plt.subplot(1, 2, 2)
	plt.imshow(impulse_response)
	plt.show()

	charge = deconvolve(tracks, imp)

	charge = np.roll(charge, offset, axis=(2, 3))

	for i in range(len(tracks)):
		plot_track(tracks[i], row=1, num_rows=2, show=False)
		plot_track(charge[i], row=2, num_rows=2, show=False)
		plt.show()

		if i == num_plots:
			break

	save_dataset(new_dataset, charge.astype('float32'), info_sets, -1)