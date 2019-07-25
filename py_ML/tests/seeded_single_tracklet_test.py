import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet
from py_datatools.plotting_tools import *
import matplotlib.pyplot as plt

tf.enable_eager_execution()

train_tracks, infosets = dt.load_whole_named_dataset('6_tracklets_large_calib_train')

train_tracklets = train_tracks.reshape((-1, 17, 24, 1))
infosets = np.repeat(infosets, 6, axis=0)

randomize = np.arange(len(train_tracklets))
np.random.shuffle(randomize)

train_tracklets = train_tracklets[randomize]
infosets = infosets[randomize]
train_labels = infosets[:, 0]

pid_model = cmod.VeryComplexConvTrackletPID()

pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
				  loss='binary_crossentropy',
				  metrics=['accuracy'],#, cmet.PionEfficiencyAtElectronEfficiency(0.9)],
				  )

pid_model.fit(train_tracklets,
			  train_labels,
			  batch_size=512,
			  epochs=100,
			  validation_split=0.2,)
			  # validation_data=(test_tracklets, test_labels),)

l = pid_model.layers[0].layers[0]
deconv = l(train_tracklets)

plt.subplot(2, 1, 1)
plt.imshow(train_tracklets[1,:,:,0])
plt.subplot(2, 1, 2)
plt.imshow(deconv[1,:,:,0])
plt.show()

# units = pid_model.get_weights()[0]
# u1 = units[:,:,0,0]
# u2 = units[:,:,0,1]
#
# rotate_units(u1, u2, vmin=-1.5, vmax=1.5)
#
# thetas = np.linspace(0,np.pi,20)
#
# plot_conv_units(pid_model.get_weights()[0], vmin=None, vmax=None)