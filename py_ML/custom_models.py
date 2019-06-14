import tensorflow as tf


class SuperModel(tf.keras.Model):
	def save(self, path):
		self.save_weights(path, save_format='tf')

	def load(self, path):
		self.load_weights(path)

class TrackletModelMultiplexer(SuperModel):
	def __init__(self, tracklet_model):
		super(TrackletModelMultiplexer, self).__init__()
		self.tracklet_model = tracklet_model

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True, input_shape=(6,)),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def call(self, tracks):
		return self.ann_model(tf.transpose(tf.map_fn(self.tracklet_model, tf.transpose(tracks, (1, 0, 2, 3, 4))), (1, 0, 2))[:,:,0])

class SimpleSingleTrackletConvPID(SuperModel):
	def __init__(self, num_filters, kernel_size=(17, 24), use_bias=False):
		super(SimpleSingleTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=1, use_bias=use_bias,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.6),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class ComplexConvTrackletPID(SuperModel):
	def __init__(self):
		super(ComplexConvTrackletPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(3, 4), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(3, 4), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class FullTrackletConvPID(SuperModel):
	def __init__(self):
		super(FullTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 24), strides=[17,24], use_bias=False,
								   bias_initializer='normal',
								   activation='sigmoid', padding='valid', ),
			tf.keras.layers.Flatten(),
		])

	def call(self, tracklets):
		return self.conv_model(tracklets)

class PartialTrackletConvPID(SuperModel):
	def __init__(self):
		super(PartialTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=5, kernel_size=(17, 3), strides=[1,1], use_bias=False,
								   bias_initializer='normal',
								   activation='relu', padding='valid', ),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.4),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))