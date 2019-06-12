import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
import matplotlib.pyplot as plt
import os

train_tracks, train_infosets = dt.load_whole_named_dataset('6_tracklets_large_train')
train_tracks = np.expand_dims(train_tracks, axis=-1)
train_labels = train_infosets[:, 0]

test_tracks, test_infosets = dt.load_whole_named_dataset('6_tracklets_large_test')
test_tracks = np.expand_dims(test_tracks, axis=-1)
test_labels = test_infosets[:, 0]

for i in range(100):
	test_tracks /= np.sum(test_tracks, axis=(2,3), keepdims=True)
	train_tracks /= np.sum(train_tracks, axis=(2,3), keepdims=True)

	tracklet_pid_model = cmod.FullTrackletConvPID()
	track_pid_model = cmod.TrackletModelMultiplexer(tracklet_pid_model)

	track_pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
					  loss='binary_crossentropy',
					  metrics=['accuracy', cmet.PionEfficiencyAtElectronEfficiency(0.9)],
					  )

	history = track_pid_model.fit(train_tracks,
				  train_labels,
				  batch_size=512,
				  epochs=50,
				  validation_data=(test_tracks, test_labels),)

	conv_unit = track_pid_model.get_weights()[0][:,:,0,0]

	# plt.imshow(conv_unit)
	# plt.show()

	new_train_tracks = dt.project_tracklet_out_of_dataset(train_tracks, conv_unit).reshape((-1, 6, 17, 24))
	new_test_tracks = dt.project_tracklet_out_of_dataset(test_tracks, conv_unit).reshape((-1, 6, 17, 24))

	dt.save_dataset('recursive_train', new_train_tracks, train_infosets, -1)
	dt.save_dataset('recursive_test', new_test_tracks, test_infosets, -1)

	np.save(os.path.dirname(__file__) + '/recursive_units/unit_%d.npy' % i, conv_unit)

	train_tracks, train_infosets = dt.load_whole_named_dataset('recursive_train')
	train_tracks = np.expand_dims(train_tracks, axis=-1)
	train_labels = train_infosets[:, 0]

	test_tracks, test_infosets = dt.load_whole_named_dataset('recursive_test')
	test_tracks = np.expand_dims(test_tracks, axis=-1)
	test_labels = test_infosets[:, 0]