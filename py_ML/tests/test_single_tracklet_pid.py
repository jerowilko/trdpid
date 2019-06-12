import tensorflow as tf
import numpy as np
import py_datatools.datatools as dt
import py_ML.custom_models as cmod
import py_ML.custom_metrics as cmet

train_tracks, infosets = dt.load_whole_named_dataset('medium_train')
train_tracklets = train_tracks.reshape((-1, 17, 24, 1))
train_labels = np.repeat(infosets[:, 0], 6)

test_tracks, infosets = dt.load_whole_named_dataset('medium_test')
test_tracklets = test_tracks.reshape((-1, 17, 24, 1))
test_labels = np.repeat(infosets[:, 0], 6)

pid_model = cmod.SimpleSingleTrackletConvPID(10, kernel_size=(3, 2), use_bias=True)

pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
				  loss='binary_crossentropy',
				  metrics=['accuracy', cmet.PionEfficiencyAtElectronEfficiency(0.9)],
				  shuffle=True
				  )

pid_model.fit(train_tracklets,
			  train_labels,
			  batch_size=32,
			  epochs=100,
			  validation_data=(test_tracklets, test_labels), )
