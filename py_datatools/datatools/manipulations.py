import tensorflow as tf
import numpy as np

def project_tracklet_out_of_dataset(tracks, tracklet):
	t = np.expand_dims(np.expand_dims(tracklet, axis=0), axis=-1)
	tnorm = np.sqrt(np.sum(t**2))
	t /= tnorm
	olaps = np.sum(tracks * t, axis=(2, 3), keepdims=True)
	return tracks - t * olaps

# def deep_dream

def project_conv_unit_out_of_dataset(tracks, conv_unit, strides, mode):
	if mode == 'valid':
		num_x_strides = (tracks[0, 0].shape[0] - conv_unit.shape[0]) // strides[0]
		num_y_strides = (tracks[0, 0].shape[1] - conv_unit.shape[1]) // strides[1]
	elif mode == 'same':
		num_x_strides = tracks[0, 0].shape[0] // strides[0]
		num_y_strides = tracks[0, 0].shape[1] // strides[1]

	for i in range(num_x_strides):
		for j in range(num_y_strides):
			tracklet = np.expand_dims(np.zeros_like(tracks[0, 0]), axis=0)

			i_start = i*strides[0]
			i_end = min(i*strides[0] + conv_unit.shape[0], tracklet.shape[0])

			j_start = j*strides[1]
			j_end = min(j*strides[1] + conv_unit.shape[1], tracklet.shape[1])

			tracklet[i_start: i_end, j_start: j_end] = conv_unit[0,:i_end - i_start, :j_end - j_start]
			tracks = project_tracklet_out_of_dataset(tracks, tracklet)

	return tracks