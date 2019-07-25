import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
from py_datatools.plotting_tools import *

train_tracks, train_infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_train')
train_tracks = np.expand_dims(train_tracks, axis=-1)
train_labels = train_infosets[:, 0]

tracklet_pid_model = cmod.ComplexConvTrackletPID()
track_pid_model = cmod.TrackletModelMultiplexer(tracklet_pid_model)

track_pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
				  loss='binary_crossentropy',
				  metrics=['accuracy', cmet.PionEfficiencyAtElectronEfficiency(0.9)],
				  )

history = track_pid_model.fit(train_tracks,
			  train_labels,
			  batch_size=512,
			  epochs=100,
			  validation_split=0.2,)
			  # validation_data=(test_tracks, test_labels),)

train_logits = track_pid_model.predict(train_tracks)

# misslabelled_train_electron_map = np.logical_and(train_labels==1.0, (train_logits < 0.5)[:,0])
#
# plot_conv_units(tracklet_pid_model.get_weights()[0])

bins = np.linspace(-0.001, 1.001, 1002)

vals, _ = np.histogram(train_logits[train_labels==1], bins=bins)
plt.plot(bins[1:] - bins[0] / 2, vals, color='b')

vals, _ = np.histogram(train_logits[train_labels==0], bins=bins)
plt.plot(bins[1:] - bins[0] / 2, vals, color='r')

plt.show()