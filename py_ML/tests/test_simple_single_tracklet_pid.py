import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
from py_datatools.plotting_tools import *

train_tracks, infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_deconvoluted_train')
train_tracklets = train_tracks[:100000].reshape((-1, 17, 24, 1))
train_labels = np.repeat(infosets[:100000][:, 0], 6)

# test_tracks, infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_test')
# test_tracklets = test_tracks.reshape((-1, 17, 24, 1))
# test_labels = np.repeat(infosets[:, 0], 6)

pid_model = cmod.SimpleSingleTrackletConvPID(2, kernel_size=(17, 24), use_bias=True)

pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
				  loss='binary_crossentropy',
				  metrics=['accuracy'],#, cmet.PionEfficiencyAtElectronEfficiency(0.9)],
				  )

pid_model.fit(train_tracklets,
			  train_labels,
			  batch_size=32,
			  epochs=100,
			  validation_split=0.2)
			  # validation_data=(test_tracklets, test_labels),)

units = pid_model.get_weights()[0]
u1 = units[:,:,0,0]
u2 = units[:,:,0,1]

rotate_units(u1, u2, vmin=-1.5, vmax=1.5)

thetas = np.linspace(0,np.pi,20)

for t in thetas:
	u = np.cos(t) * u1 + np.sin(t) * u2
	plt.imshow(u)
	plt.colorbar()
	plt.show()

plot_conv_units(pid_model.get_weights()[0], vmin=None, vmax=None)