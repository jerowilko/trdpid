import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
from py_datatools.plotting_tools import *
import matplotlib.pyplot as plt

train_tracks, train_infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_train')

train_tracks = train_tracks[train_infosets[:,0]==0]
train_infosets = train_infosets[train_infosets[:,0]==0]

train_tracks = np.expand_dims(train_tracks, axis=-1)
train_Ps = train_infosets[:, 5]

tracklet_momentum_model = cmod.ComplexConvTrackletMomentumModel()
track_momentum_model = cmod.TrackletMomentumModelMultiplexer(tracklet_momentum_model)

track_momentum_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
				  loss='mean_squared_error',
				  )

plt.hist(train_Ps, bins=1000)
plt.show()

history = track_momentum_model.fit(train_tracks,
              train_Ps,
			  batch_size=512,
			  epochs=100,
			  validation_split=0.2,)
			  # validation_data=(test_tracks, test_labels),)

p_Ps = track_momentum_model.predict(train_tracks)

vals, bins, _ = plt.hist(train_Ps, bins=1000)
plt.hist(p_Ps, bins=bins)
plt.show()

# misslabelled_train_electron_map = np.logical_and(train_labels==1.0, (train_logits < 0.5)[:,0])
#
# plot_conv_units(tracklet_momentum_model.get_weights()[0])

bins = np.linspace(-0.001, 1.001, 1002)

vals, _ = np.histogram(train_logits[train_labels==1], bins=bins)
plt.plot(bins[1:] - bins[0] / 2, vals, color='b')

vals, _ = np.histogram(train_logits[train_labels==0], bins=bins)
plt.plot(bins[1:] - bins[0] / 2, vals, color='r')

plt.show()