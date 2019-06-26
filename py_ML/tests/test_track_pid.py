import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
from py_datatools.plotting_tools import *

train_tracks, train_infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_deconvoluted_train')
train_tracks = np.expand_dims(train_tracks, axis=-1)
train_labels = train_infosets[:, 0]

test_tracks, test_infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_deconvoluted_test')
test_tracks = np.expand_dims(test_tracks, axis=-1)
test_labels = test_infosets[:, 0]

tracklet_pid_model = cmod.PartialTrackletConvPID()
track_pid_model = cmod.TrackletModelMultiplexer(tracklet_pid_model)

track_pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
				  loss='binary_crossentropy',
				  metrics=['accuracy', cmet.PionEfficiencyAtElectronEfficiency(0.9)],
				  )

history = track_pid_model.fit(train_tracks,
			  train_labels,
			  batch_size=512,
			  epochs=10000,
			  validation_data=(test_tracks, test_labels),)

train_logits = track_pid_model.predict(train_tracks)

misslabelled_train_electron_map = np.logical_and(train_labels==1.0, (train_logits < 0.5)[:,0])

plot_conv_units(tracklet_pid_model.get_weights()[0])