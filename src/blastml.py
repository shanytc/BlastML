import os
import shutil
from PIL import Image
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, BatchNormalization, LeakyReLU, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import json


# load image using Pillow
def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img.load()
		img = img.convert('RGB')
	return img

#
# Configurations:
#
# Activations
# ============
# 'softmax'
# 'relu'
# 'sigmoid'
# 'elu'
# 'selu'
# 'softplus'
# 'softsign'
# 'tanh'
# 'hard_sigmoid'
# 'exponential'
# 'linear'

# Optimizers:
# =============
# 'RMSprop' / optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# 'Adagrad'
# 'Adadelta'
# 'Adam'
# 'Adamax'
# 'Nadam'
# 'SGD' / optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 'adam' / optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Loss Functions
# ==============
# 'mean_squared_error'
# 'mean_absolute_error'
# 'mean_absolute_percentage_error'
# 'mean_squared_logarithmic_error'
# 'squared_hinge'
# 'hinge'
# 'categorical_hinge'
# 'logcosh'
# 'categorical_crossentropy'
# 'sparse_categorical_crossentropy'
# 'binary_crossentropy'
# 'kullback_leibler_divergence'
# 'poisson'
# 'cosine_proximity'

# Class Modes
# ============
# categorical
# binary
# sparse


class CFG:
	def __init__(self, image={}, augmentation={}, hyper_params={}, multithreading={}, dataset={}, model={}):
		self.input_image_width = image['width'],
		self.input_image_height = image['height'],
		self.input_image_channels = image['channels'],
		self.batch = hyper_params['batch']
		self.epochs = hyper_params['epochs']
		self.optimizer = hyper_params['optimizer']
		self.loss_function = hyper_params['loss_function']
		self.num_classes = hyper_params['classes']
		self.class_mode = hyper_params['class_mode']
		self.compile_metrics = hyper_params['compile_metrics']
		self.enable_multithreading = multithreading['enable_multithreading']
		self.threads = multithreading['threads']
		self.train_path = dataset['train_path']
		self.validation_path = dataset['validation_path']
		self.infer_path = dataset['infer_path']
		self.model_output_path = model['model_output_path']
		self.model_name = model['model_name']
		self.save_model = model['save_model']
		self.save_weights = model['save_weights']
		self.save_history = model['save_history']
		self.enable_saving = model['enable_saving']
		self.reset_learn_phase = model['reset_learn_phase']
		self.load_model_embeddings = model['load_model_embeddings']
		self.augmentation = augmentation

	def is_augmentation_enagled(self):
		return self.is_augment_enabled

	def set_optimizer(self, optimizer=None):
		if optimizer is None:
			return self

		self.optimizer = optimizer

		return self

	def get_width(self):
		return self.input_image_width[0]

	def get_height(self):
		return self.input_image_height[0]

	def get_channels(self):
		return self.input_image_channels[0]

	def get_model_name(self):
		return self.model_name

	def get_model_output_path(self):
		if self.model_output_path[-1] == '/':
			return self.model_output_path

		return self.model_output_path + '/'

	def get_should_save_history(self):
		return self.save_history

	def get_should_save_weights(self):
		return self.save_weights

	def get_should_save_model(self):
		return self.save_model

	def get_train_path(self):
		return self.train_path

	def get_validation_path(self):
		return self.validation_path

	def get_infer_path(self):
		return self.infer_path

	def get_num_epochs(self):
		return self.epochs

	def get_batch_size(self):
		return self.batch

	def get_num_classes(self):
		return self.num_classes

	def get_class_mode(self):
		return self.class_mode

	def get_num_threads(self):
		return self.threads

	def get_multithreading_status(self):
		return self.enable_multithreading

	def get_optimizer(self):
		return self.optimizer

	def get_loss_function(self):
		return self.loss_function

	def get_compile_metrics(self):
		return self.compile_metrics


class BlastML:
	def __init__(self, cfg=None):
		self.config = cfg

		if cfg is None:
			self.config = CFG()

		self.model = None
		self.predictions = None
		self.classes = None
		self.history = None


	def resnet18(self):
		self.create() \
		.add_2d(filters=64, kernel=(7, 7), strides=(2, 2), activation='relu', padding='same', input_shape=(self.config.get_width(), self.config.get_height(), self.config.get_channels())) \
		.add_batch_normalize(axis=3) \
		.add_max_pooling(size=(3, 3), strides=(2, 2), padding='same') \
		.add_2d(filters=64, kernel=(3, 3), strides=(1, 1),  activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=64, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=64, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=64, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=128, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=128, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_2d(filters=128, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=128, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=128, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=256, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=256, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_2d(filters=256, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=256, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=256, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=512, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=512, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_2d(filters=512, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=512, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_2d(filters=512, kernel=(3, 3), strides=(1, 1), activation='relu', padding='same') \
		.add_batch_normalize(axis=3) \
		.add_average_pooling(size=(1, 1), strides=(1, 1)) \
		.add_flatten() \
		.add_dense(size=self.config.get_num_classes(), activation='softmax', name="layer_classes") \
		.show_model_summary()

		return self

	def vgg16(self):
		self.create()\
		.add_zero_padding(input_shape=(self.config.get_width(), self.config.get_height(), self.config.get_channels()))\
		.add_2d(filters=64, kernel=(3, 3), activation="relu") \
		.add_zero_padding() \
		.add_2d(filters=64, kernel=(3, 3), activation='relu')\
		.add_max_pooling(strides=(2, 2))\
		.add_zero_padding() \
		.add_2d(filters=128, kernel=(3, 3), activation="relu") \
		.add_zero_padding() \
		.add_2d(filters=128, kernel=(3, 3), activation='relu') \
		.add_max_pooling(strides=(2, 2)) \
		.add_zero_padding() \
		.add_2d(filters=256, kernel=(3, 3), activation="relu") \
		.add_zero_padding() \
		.add_2d(filters=256, kernel=(3, 3), activation='relu') \
		.add_zero_padding() \
		.add_2d(filters=256, kernel=(3, 3), activation='relu') \
		.add_max_pooling(strides=(2, 2)) \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation="relu") \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation='relu') \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation='relu') \
		.add_max_pooling(strides=(2, 2)) \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation="relu") \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation='relu') \
		.add_zero_padding() \
		.add_2d(filters=512, kernel=(3, 3), activation='relu') \
		.add_max_pooling(strides=(2, 2)) \
		.add_flatten()\
		.add_dense(size=4096, activation='relu', name="layer_features") \
		.add_dropout(dropout=0.5) \
		.add_dense(size=4096, activation='relu', name="layer_features2") \
		.add_dropout(dropout=0.5) \
		.add_dense(size=self.config.get_num_classes(), activation='softmax', name="layer_classes")\
		.show_model_summary()

		return self

	def simple(self):

		""" use of 6 conv layers, 1 fully connected """
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(self.config.get_width(), self.config.get_height(), self.config.get_channels())))
		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))  # 64 filters, size 3x3
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 max pooling
		model.add(Dropout(0.25))  # technique used to tackle Overfitting

		model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())  # flat
		model.add(Dense(512, activation='relu', name="layer_features"))
		model.add(Dropout(0.5))
		model.add(Dense(size=self.config.get_num_classes(), activation='softmax', name="layer_classes"))  # number of classes

		model.summary()
		self.model = model

		return self

	def create(self):
		self.model = None
		self.model = Sequential()
		return self

	def add_2d(self, filters=32, kernel=(3, 3), **kwargs):
		# stride (1,1)
		self.model.add(Conv2D(filters, kernel, **kwargs))  # 32 filters, size of 3x3
		return self

	def add_zero_padding(self, padding=(1, 1), **kwargs):
		self.model.add(ZeroPadding2D(padding=padding, **kwargs))
		return self

	def add_leaky_relu(self):
		self.model.add(LeakyReLU())
		return self

	def add_batch_normalize(self, **kwargs):
		self.model.add(BatchNormalization(**kwargs))
		return self

	def add_max_pooling(self, size=(2, 2), strides=None, **kwargs):
		self.model.add(MaxPooling2D(pool_size=size, strides=strides, **kwargs))  # 2x2 max pooling
		return self

	def add_average_pooling(self, size=(2, 2), strides=None, **kwargs):
		self.model.add(AveragePooling2D(pool_size=size, strides=strides, **kwargs))
		return self

	def add_dropout(self, dropout=0.25):
		self.model.add(Dropout(dropout))
		return self

	def add_flatten(self):
		self.model.add(Flatten())
		return self

	def add_dense(self, size=512, **kwargs):
		self.model.add(Dense(size, **kwargs))
		return self

	def add_basic_block(self, filters=64, kernel=(3, 3), activation='relu'):
		self.add_2d(filters=filters, kernel=kernel, padding='same', activation=activation) \
			.add_2d(filters=filters, kernel=kernel, activation=activation) \
			.add_max_pooling() \
			.add_dropout()

		return self

	def load_model(self):
		# prepare json for embeddings only
		if self.config.load_model_embeddings is True:
			with open(self.config.get_model_output_path() + self.config.get_model_name() + '.json', 'r') as f:
				json_file = json.load(f)

			config = json_file['config']
			layers = config['layers']
			for layer in layers:
				if layer['class_name'] == 'Dropout':  # remove dropout (better embeddings)
					del layer['config']
					del layer['class_name']
				elif layer['class_name'] == 'Dense':  # remove dense layer (100 classes) keep 512 embeddings only
					layer_config = layer['config']
					name = layer_config['name']
					if name == 'layer_classes':
						del layer['config']
						del layer['class_name']
			while {} in layers:
				layers.remove({})

			model_json = json.dumps(json_file)
		else:
			json_file = open(self.config.get_model_output_path() + self.config.get_model_name() + '.json', 'r')
			model_json = json_file.read()

		if self.config.reset_learn_phase:
			K.set_learning_phase(0)
		model = model_from_json(model_json)
		model.load_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".h5", by_name=True)

		self.model = model

		# load history
		with open(self.config.get_model_output_path() + self.config.get_model_name() + '.history', 'rb') as f:
			self.history = pickle.load(f)

		return self

	def export_to_tf(self):
		if self.model is None:
			print("No model found to export")
			return self
		#
		# Export Generates 4 files:
		# -------------------------
		# checkpoint defines the model checkpoint path which is "tf_model" in our case.
		# .meta stores the graph structure,
		# .data stores the values of each variable in the graph
		# .index identifies the checkpoint.
		#

		saver = tf.train.Saver()
		sess = K.get_session()
		saver.save(sess, self.config.get_model_output_path() + self.config.get_model_name() + ".tf")
		fw = tf.summary.FileWriter('logs', sess.graph)
		fw.close()
		return self

	def show_model_summary(self):
		self.model.summary()
		return self

	def freeze_model_layers(self, num_of_layers):
		if self.model is None:
			return self

		for layer in self.model.layers[:num_of_layers]:
			layer.trainable = False

		return self

	def unfreeze_model_layers(self):
		if self.model is None:
			return self

		for layer in self.model.layers:
			layer.trainable = True

		return self

	def get_model(self):
		return self.model

	def load_classes(self, dirpath=''):
		if dirpath == '':
			self.classes = []
			return self

		self.classes = []
		for subdir, dirs, files in os.walk(dirpath):
			for filename in files:
				self.classes.append(subdir.replace(dirpath+'/', ''))

		return self

	def compile(self):
		if self.model is None:
			return self

		self.model.compile(optimizer=self.config.get_optimizer(), loss=self.config.get_loss_function(), metrics=self.config.get_compile_metrics())

		return self

	def train(self):
		if self.model is None:
			return self

		if self.config.get_train_path() == '':
			return self


		# default with no augmentation (except rescale)
		train_datagen = ImageDataGenerator(rescale=1./255)
		valid_datagen = ImageDataGenerator(rescale=1./255)

		# use customised augmentation settings
		if self.config.augmentation['train']['enable']:
			train_datagen = ImageDataGenerator(
				rescale=self.config.augmentation['train']['rescale'],
				samplewise_center=self.config.augmentation['train']['samplewise_center'],
				featurewise_std_normalization=self.config.augmentation['train']['featurewise_std_normalization'],
				samplewise_std_normalization=self.config.augmentation['train']['samplewise_std_normalization'],
				rotation_range=self.config.augmentation['train']['rotation_range'],
				width_shift_range=self.config.augmentation['train']['width_shift_range'],
				height_shift_range=self.config.augmentation['train']['height_shift_range'],
				brightness_range=self.config.augmentation['train']['brightness_range'],
				shear_range=self.config.augmentation['train']['shear_range'],
				zoom_range=self.config.augmentation['train']['zoom_range'],
				channel_shift_range=self.config.augmentation['train']['channel_shift_range'],
				fill_mode=self.config.augmentation['train']['fill_mode'],
				horizontal_flip=self.config.augmentation['train']['horizontal_flip'],
				vertical_flip=self.config.augmentation['train']['vertical_flip'],
			)

		if self.config.augmentation['validation']['enable']:
			valid_datagen = ImageDataGenerator(
				rescale=self.config.augmentation['validation']['rescale'],
				samplewise_center=self.config.augmentation['validation']['samplewise_center'],
				featurewise_std_normalization=self.config.augmentation['validation']['featurewise_std_normalization'],
				samplewise_std_normalization=self.config.augmentation['validation']['samplewise_std_normalization'],
				rotation_range=self.config.augmentation['validation']['rotation_range'],
				width_shift_range=self.config.augmentation['validation']['width_shift_range'],
				height_shift_range=self.config.augmentation['validation']['height_shift_range'],
				brightness_range=self.config.augmentation['validation']['brightness_range'],
				shear_range=self.config.augmentation['validation']['shear_range'],
				zoom_range=self.config.augmentation['validation']['zoom_range'],
				channel_shift_range=self.config.augmentation['validation']['channel_shift_range'],
				fill_mode=self.config.augmentation['validation']['fill_mode'],
				horizontal_flip=self.config.augmentation['validation']['horizontal_flip'],
				vertical_flip=self.config.augmentation['validation']['vertical_flip'],
			)

		train_generator = train_datagen.flow_from_directory(
			self.config.get_train_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode())

		val_generator = valid_datagen.flow_from_directory(
			self.config.get_validation_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode())

		# steps for training
		steps = train_generator.n/train_generator.batch_size

		# store all training information while training the dataset
		self.history = History()

		print("Training network...")
		self.model.fit_generator(
			train_generator,
			steps_per_epoch=steps,
			epochs=self.config.get_num_epochs(),
			validation_data=val_generator,
			validation_steps=self.config.get_batch_size(),
			use_multiprocessing=self.config.get_multithreading_status(),
			workers=self.config.get_num_threads(),
			callbacks=[self.history])

		# Check for saving flag
		if self.config.enable_saving is True:

			# remove old model when all saving features are enabled
			if self.config.get_should_save_model() and self.config.get_should_save_weights() and self.config.get_should_save_history():
				print("Removing old Model...")
				# remove content of output folder (clear model data)
				for the_file in os.listdir(self.config.get_model_output_path()):
					file_path = os.path.join(self.config.get_model_output_path(), the_file)
					try:
						if os.path.isfile(file_path):
							os.unlink(file_path)
					except Exception as e:
						print(e)

			if self.config.get_should_save_model() is True and self.config.get_model_name() != '':
				# save model and weights
				model_json = self.model.to_json()
				with open(self.config.get_model_output_path() + self.config.get_model_name() + ".json", "w") as json_file:
					json_file.write(model_json)

				print("Model saved to disk.")

			if self.config.get_should_save_weights() is True and self.config.get_model_name() != '':
				# serialize weights to HDF5
				self.model.save_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".h5")
				print("Model's weights saved to disk.")

			if self.config.get_should_save_history() is True and self.config.get_model_name() != '':
				with open(self.config.get_model_output_path() + self.config.get_model_name() + ".history", "wb") as f:
					pickle.dump(self.history.history, f)
				print("Model's history saved.")

		return self

	def evaluate(self):
		if self.model is None:
			return self

		if self.config.get_infer_path() == '':
			return self

		test_datagen = ImageDataGenerator(rescale=1./255)

		test_generator = test_datagen.flow_from_directory(
			directory=self.config.get_infer_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			color_mode="rgb",
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode(),
			shuffle=False
		)

		steps = test_generator.n/test_generator.batch_size

		print("Evaluating network...")
		score = self.model.evaluate_generator(generator=test_generator, steps=steps, use_multiprocessing=self.config.get_multithreading_status(), workers=self.config.get_num_threads())

		evaluation = zip(self.model.metrics_names, score)
		print("Network Evaluation:")
		for name, output in evaluation:
			print(name + ": " + str(output))

		# print("Testing network...")
		# test_generator.reset()
		# self.predictions = self.model.predict_generator(
		# 	test_generator,
		# 	steps=steps,
		# 	use_multiprocessing=self.config.get_multithreading_status(),
		# 	workers=self.config.get_num_threads(),
		# 	verbose=1)

		return self

	def infer(self, image_path=None):
		if self.model is None:
			return self

		if image_path is not None and image_path != '':
			self.config.infer_path = image_path
		else:
			if self.config.get_infer_path() == '':
				return self

		train_datagen = ImageDataGenerator(rescale=1./255)
		train_generator = train_datagen.flow_from_directory(
			self.config.get_train_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode())

		infer_datagen = ImageDataGenerator(rescale=1./255)
		infer_generator = infer_datagen.flow_from_directory(
			directory=self.config.get_infer_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			color_mode="rgb",
			batch_size=self.config.get_batch_size(),
			class_mode=None,
			shuffle=False
		)

		filenames = infer_generator.filenames
		steps = infer_generator.n/infer_generator.batch_size

		print("Infering folder(s)...")
		infer_generator.reset()
		embeddings = self.model.predict_generator(infer_generator, steps=steps, use_multiprocessing=self.config.get_multithreading_status(), workers=self.config.get_num_threads(), verbose=1)

		# get classes
		if self.config.load_model_embeddings is False:
			predicted_class_indices = np.argmax(embeddings, axis=1)  # get index of classes
			labels = train_generator.class_indices
			labels = dict((v, k) for k,v in labels.items())
			pred_classes = [labels[k] for k in predicted_class_indices]

			# get embeddings
			data = list(zip(filenames, embeddings, pred_classes))
		else:
			data = list(zip(filenames, embeddings))

		return data

	def plot_history(self):
		if self.history is None:
			return self

		print("Ploting history to images...")

		# summarize history for accuracy
		fig, ax = plt.subplots()
		ax.grid(True)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_xlim(0, len(self.history['acc']))
		list = []
		for i in range(0, len(self.history['acc'])):
			list.append(i+1)
		ax.set_xticklabels(list)

		plt.plot(self.history['acc'])
		plt.plot(self.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train (acc)', 'test (val_acc)'], loc='upper left')
		plt.show()
		plt.savefig(self.config.get_model_output_path() + self.config.get_model_name() + ".accuracy.png")
		plt.clf()

		# summarize history for loss
		fig, ax = plt.subplots()
		ax.grid(True)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_xlim(0, len(self.history['loss']))
		list = []
		for i in range(0, len(self.history['loss'])):
			list.append(i+1)
		ax.set_xticklabels(list)

		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.history['loss'])
		plt.plot(self.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train (loss)', 'test (val_loss)'], loc='upper left')
		plt.savefig(self.config.get_model_output_path() + self.config.get_model_name() + ".loss.png")
		plt.show()

		return self

	def get_history(self):
		return self.history

	def get_predictions(self):
		return self.predictions

	def get_predictions_indexes(self):
		return np.argmax(self.predictions, axis=1)


def main():
	# Create a configuration settings for BlastML
	cfg = CFG(
			image={
				'width': 224,
				'height': 224,
				'channels': 3
			},
			augmentation={
				'train': {
					'enable': True,
					'featurewise_center': False,
					'samplewise_center': False,
					'featurewise_std_normalization': False,
					'samplewise_std_normalization': False,
					'rotation_range': 0,
					'width_shift_range': 0.0,
					'height_shift_range': 0.0,
					'brightness_range': None,
					'shear_range': 0.2,
					'zoom_range': 0.2,
					'channel_shift_range': 0.0,
					'fill_mode': 'nearest',
					'horizontal_flip': True,
					'vertical_flip': True,
					'rescale': 1./255
				},
				'validation': {
					'enable': True,
					'featurewise_center': False,
					'samplewise_center': False,
					'featurewise_std_normalization': False,
					'samplewise_std_normalization': False,
					'rotation_range': 0,
					'width_shift_range': 0.0,
					'height_shift_range': 0.0,
					'brightness_range': None,
					'shear_range': 0.2,
					'zoom_range': 0.2,
					'channel_shift_range': 0.0,
					'fill_mode': 'nearest',
					'horizontal_flip': True,
					'vertical_flip': True,
					'rescale': 1./255
				}
			},
			hyper_params={
				'batch': 16,
				'epochs': 10,
				'classes': 5,
				'class_mode': 'sparse',
				'optimizer': 'adam',
				'loss_function': 'sparse_categorical_crossentropy',
				'compile_metrics': ['accuracy']
			},
			multithreading={
				'enable_multithreading': True,
				'threads': 5
			},
			dataset={
				'train_path': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/train',
				'validation_path': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/validation',
				'infer_path': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/inference',
			},
			model={
				'model_output_path': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/model',
				'model_name': 'shanynet',
				'load_model_embeddings': False,  # strip dropouts and fc layers
				'enable_saving': False,
				'save_model': True,
				'save_weights': True,
				'save_history': True,
				'reset_learn_phase': True
			})

	# Create a BlastML instance
	net = BlastML(cfg=cfg)

	# Create a very simple CNN instance
	# Net.simple().compile().train().evaluate().infer()

	# create a vgg16 instance
	net.vgg16().compile().train().evaluate()

	#  create a resnet18 instance
	# net.resnet18().compile().train().evaluate()

	# Create a custom CNN instance
	# net.create()\
	# 	.add_2d(filters=32, kernel=(3, 3), activation="relu", padding='same', input_shape=(net.config.get_width(), net.config.get_height(), net.config.get_channels()))\
	# 	.add_2d(filters=32, kernel=(3, 3), activation='relu')\
	# 	.add_max_pooling()\
	# 	.add_dropout()\
	# 	.add_basic_block()\
	# 	.add_basic_block()\
	# 	.add_flatten()\
	# 	.add_dense(size=512, activation='relu', name="layer_features")\
	# 	.add_dense(size=cfg.get_num_classes(), activation='softmax', name="layer_classes")\
	# 	.show_model_summary()\
	# 	.compile()\
	# 	.train()\
	# 	.evaluate()

	# use the code below to load model, create history (optional) and infer (test) your files
	cfg.threads = 1  # better to use 1 thread, but you can change it.
	res = net.load_model()#.export_to_tf().plot_history().infer()
	# print(res)  # show embeddings/classification results

main()