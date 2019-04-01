import os
from os import listdir, makedirs
from os.path import isfile, join
import io
from termcolor import colored
import shutil
import configparser
from collections import defaultdict
from PIL import Image
import numpy as np
from keras.models import Model
from keras.models import Sequential, model_from_json
from keras.layers import UpSampling2D, Concatenate, Add, Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, BatchNormalization, LeakyReLU, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
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
	def __init__(self, project={}, image={}, augmentation={}, hyper_params={}, multithreading={}, model={}, darknet={}):
		# Project settings
		self.project_name = project['project_name']
		self.project_root_path = project['root']
		self.project_folder_path = self.project_root_path + project['project_folder']
		self.project_dataset_path = self.project_folder_path + project['dataset']
		self.project_train_path = self.project_folder_path + project['train']
		self.project_inference_path = self.project_folder_path + project['inference']
		self.project_validation_path = self.project_folder_path + project['validation']
		self.project_model_path = self.project_folder_path + project['model']

		# CNN Image input settings
		self.input_image_width = image['width'],
		self.input_image_height = image['height'],
		self.input_image_channels = image['channels'],

		# CNN HyperParams settings
		self.batch = hyper_params['batch']
		self.epochs = hyper_params['epochs']
		self.optimizer = hyper_params['optimizer']
		self.loss_function = hyper_params['loss_function']
		self.num_classes = hyper_params['classes']
		self.class_mode = hyper_params['class_mode']
		self.compile_metrics = hyper_params['compile_metrics']
		self.shuffle_training = hyper_params['shuffle']

		# Multithreading (Training) settings
		self.enable_multithreading = multithreading['enable_multithreading']
		self.threads = multithreading['threads']

		# Model settings
		self.model_name = self.project_name
		self.save_model = model['save_model']
		self.save_weights = model['save_weights']
		self.save_history = model['save_history']
		self.enable_saving = model['enable_saving']
		self.reset_learn_phase = model['reset_learn_phase']
		self.load_model_embeddings = model['load_model_embeddings']
		self.save_bottleneck_features = model['save_bottleneck_features']

		# Augmentation settings (Training)
		self.augmentation = augmentation

		# DarkNet settings (Object Detection)
		self.darknet_enable_saving = darknet['enable_saving']
		self.darknet_cfg = darknet['cfg']
		self.darknet_weights = darknet['weights']
		self.darknet_save_model = darknet['save_model']
		self.darknet_save_weight = darknet['save_weights']
		self.darknet_taining_data = darknet['taining_data']

	def get_project_name(self):
		return self.project_name

	def get_project_root_path(self):
		return self.project_root_path

	def get_project_folder_path(self):
		return self.project_folder_path

	def get_project_dataset_path(self):
		return self.project_dataset_path

	def get_project_train_path(self):
		return self.project_train_path

	def get_project_inference_path(self):
		return self.project_inference_path

	def get_project_validation_path(self):
		return self.project_validation_path

	def get_project_model_path(self):
		return self.project_model_path

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
		if self.project_model_path[-1] == '/':
			return self.project_model_path

		return self.project_model_path + '/'

	def get_should_save_history(self):
		return self.save_history

	def get_should_save_weights(self):
		return self.save_weights

	def get_should_save_model(self):
		return self.save_model

	def get_should_save_bottleneck_features(self):
		return self.save_bottleneck_features

	def get_train_path(self):
		return self.project_train_path

	def get_validation_path(self):
		return self.project_validation_path

	def get_infer_path(self):
		return self.project_inference_path

	def get_num_epochs(self):
		return self.epochs

	def get_batch_size(self):
		return self.batch

	def get_num_classes(self):
		return self.num_classes

	def get_shuffle_training(self):
		return self.shuffle_training

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

	def is_darknet_saving_enabled(self):
		return self.darknet_enable_saving

	def get_darknet_config(self):
		return self.darknet_cfg

	def get_darknet_weights(self):
		return self.darknet_weights

	def get_should_save_darknet_weights(self):
		return self.darknet_save_weight

	def get_should_save_darknet_model(self):
		return self.darknet_save_model

	def get_darknet_training(self):
		return self.darknet_taining_data

class BlastML:
	def __init__(self, cfg=None):
		self.config = cfg

		if cfg is None:
			self.config = CFG()

		self.model = None
		self.predictions = None
		self.classes = None
		self.history = None
		self.bottleneck = None

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

	def reset(self):
		self.model = None
		self.predictions = None
		self.classes = None
		self.history = None
		self.bottleneck = None
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
		# reset the model
		self.reset()
		model_json = None

		# prepare json for embeddings only
		if self.config.load_model_embeddings is True and os.path.isfile(self.config.get_model_output_path() + self.config.get_model_name() + '.json'):
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
			if os.path.isfile(self.config.get_model_output_path() + self.config.get_model_name() + '.json'):
				json_file = open(self.config.get_model_output_path() + self.config.get_model_name() + '.json', 'r')
				model_json = json_file.read()

		if self.config.reset_learn_phase:
			K.set_learning_phase(0)

		if model_json is not None:
			model = model_from_json(model_json)

			if os.path.isfile(self.config.get_model_output_path() + self.config.get_model_name() + '.h5'):
				model.load_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".h5", by_name=True)

			self.model = model

		if os.path.isfile(self.config.get_model_output_path() + self.config.get_model_name() + '.features'):
			self.bottleneck = np.load(open(self.config.get_model_output_path() + self.config.get_model_name() + '.features'))

		# load history
		if os.path.isfile(self.config.get_model_output_path() + self.config.get_model_name() + '.history'):
			with open(self.config.get_model_output_path() + self.config.get_model_name() + '.history', 'rb') as f:
				self.history = pickle.load(f)

		return self

	def darknet_to_keras(self):
		# test darknet files exists
		if self.config.get_darknet_config() != '' and os.path.isfile(self.config.get_darknet_config()) and self.config.get_darknet_weights() != '' and os.path.isfile(self.config.get_darknet_weights()):

			def unique_config_sections(config_file):
				#
				# Convert all config sections to have unique names.
				# Adds unique suffixes to config sections for compability with configparser.
				#
				section_counters = defaultdict(int)
				output_stream = io.StringIO()
				with open(config_file) as fin:
					for line in fin:
						if line.startswith('['):
							section = line.strip().strip('[]')
							_section = section + '_' + str(section_counters[section])
							section_counters[section] += 1
							line = line.replace(section, _section)
						output_stream.write(line)
				output_stream.seek(0)
				return output_stream

			print("converting darknet model to keras...")

			print('Loading darknet weights.')
			weights_file = open(self.config.get_darknet_weights(), 'rb')
			major, minor, revision = np.ndarray(shape=(3, ), dtype='int32', buffer=weights_file.read(12))

			if (major*10+minor) >= 2 and major < 1000 and minor < 1000:
				seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
			else:
				seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))

			print('Weights Header: ', major, minor, revision, seen)

			print('Parsing Darknet config.')
			unique_config_file = unique_config_sections(self.config.get_darknet_config())
			cfg_parser = configparser.ConfigParser()
			cfg_parser.read_file(unique_config_file)

			print('Creating Keras model.')
			input_layer = Input(shape=(None, None, 3))
			prev_layer = input_layer
			all_layers = []

			weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
			count = 0
			out_index = []
			for section in cfg_parser.sections():
				print('Parsing section {}'.format(section))
				if section.startswith('convolutional'):
					filters = int(cfg_parser[section]['filters'])
					size = int(cfg_parser[section]['size'])
					stride = int(cfg_parser[section]['stride'])
					pad = int(cfg_parser[section]['pad'])
					activation = cfg_parser[section]['activation']
					batch_normalize = 'batch_normalize' in cfg_parser[section]

					padding = 'same' if pad == 1 and stride == 1 else 'valid'

					# Setting weights.
					# Darknet serializes convolutional weights as:
					# [bias/beta, [gamma, mean, variance], conv_weights]
					prev_layer_shape = K.int_shape(prev_layer)

					weights_shape = (size, size, prev_layer_shape[-1], filters)
					darknet_w_shape = (filters, weights_shape[2], size, size)
					weights_size = np.product(weights_shape)

					print('conv2d', 'bn'
					if batch_normalize else '  ', activation, weights_shape)

					conv_bias = np.ndarray(
						shape=(filters, ),
						dtype='float32',
						buffer=weights_file.read(filters * 4))
					count += filters

					if batch_normalize:
						bn_weights = np.ndarray(
							shape=(3, filters),
							dtype='float32',
							buffer=weights_file.read(filters * 12))
						count += 3 * filters

						bn_weight_list = [
							bn_weights[0],  # scale gamma
							conv_bias,  # shift beta
							bn_weights[1],  # running mean
							bn_weights[2]  # running var
						]

					conv_weights = np.ndarray(
						shape=darknet_w_shape,
						dtype='float32',
						buffer=weights_file.read(weights_size * 4))
					count += weights_size

					# DarkNet conv_weights are serialized Caffe-style:
					# (out_dim, in_dim, height, width)
					# We would like to set these to Tensorflow order:
					# (height, width, in_dim, out_dim)
					conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
					conv_weights = [conv_weights] if batch_normalize else [
						conv_weights, conv_bias
					]

					# Handle activation.
					act_fn = None
					if activation == 'leaky':
						pass  # Add advanced activation later.
					elif activation != 'linear':
						raise ValueError(
							'Unknown activation function `{}` in section {}'.format(
								activation, section))

					# Create Conv2D layer
					if stride>1:
						# Darknet uses left and top padding instead of 'same' mode
						prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
					conv_layer = (Conv2D(
						filters, (size, size),
						strides=(stride, stride),
						kernel_regularizer=l2(weight_decay),
						use_bias=not batch_normalize,
						weights=conv_weights,
						activation=act_fn,
						padding=padding))(prev_layer)

					if batch_normalize:
						conv_layer = (BatchNormalization(
							weights=bn_weight_list))(conv_layer)
					prev_layer = conv_layer

					if activation == 'linear':
						all_layers.append(prev_layer)
					elif activation == 'leaky':
						act_layer = LeakyReLU(alpha=0.1)(prev_layer)
						prev_layer = act_layer
						all_layers.append(act_layer)

				elif section.startswith('route'):
					ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
					layers = [all_layers[i] for i in ids]
					if len(layers) > 1:
						print('Concatenating route layers:', layers)
						concatenate_layer = Concatenate()(layers)
						all_layers.append(concatenate_layer)
						prev_layer = concatenate_layer
					else:
						skip_layer = layers[0]  # only one layer to route
						all_layers.append(skip_layer)
						prev_layer = skip_layer

				elif section.startswith('maxpool'):
					size = int(cfg_parser[section]['size'])
					stride = int(cfg_parser[section]['stride'])
					all_layers.append(
						MaxPooling2D(
							pool_size=(size, size),
							strides=(stride, stride),
							padding='same')(prev_layer))
					prev_layer = all_layers[-1]

				elif section.startswith('shortcut'):
					index = int(cfg_parser[section]['from'])
					activation = cfg_parser[section]['activation']
					assert activation == 'linear', 'Only linear activation supported.'
					all_layers.append(Add()([all_layers[index], prev_layer]))
					prev_layer = all_layers[-1]

				elif section.startswith('upsample'):
					stride = int(cfg_parser[section]['stride'])
					assert stride == 2, 'Only stride=2 supported.'
					all_layers.append(UpSampling2D(stride)(prev_layer))
					prev_layer = all_layers[-1]

				elif section.startswith('yolo'):
					out_index.append(len(all_layers)-1)
					all_layers.append(None)
					prev_layer = all_layers[-1]

				elif section.startswith('net'):
					pass

				else:
					raise ValueError(
						'Unsupported section header type: {}'.format(section))

			# Create and save model.
			if len(out_index) == 0:
				out_index.append(len(all_layers)-1)

			model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
			print(model.summary())

			if self.config.is_darknet_saving_enabled():
				# remove old model when all saving features are enabled
				if self.config.get_model_name() != '' and self.config.get_should_save_darknet_model() and self.config.get_should_save_darknet_weights():
					print("Removing old darknet model...")
					darknet_files = [
						self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.json",
						self.config.get_model_output_path() + self.config.get_model_name() + '.darknet.h5'
					]
					for darknetfile in darknet_files:
						try:
							if os.path.isfile(darknetfile):
								os.unlink(darknetfile)
						except Exception as e:
							print(e)

				if self.config.get_should_save_darknet_model():
					model_json = model.to_json()
					with open(self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.json", "w") as json_file:
						json_file.write(model_json)

				if self.config.get_should_save_darknet_weights():
					model.save_weights(self.config.get_model_output_path() + self.config.get_model_name() + '.darknet.h5')
					print("Model's weights saved to disk.")

				# Check to see if all weights have been read.
				remaining_weights = len(weights_file.read()) / 4
				weights_file.close()
				print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))
				if remaining_weights > 0:
					print('Warning: {} unused weights'.format(remaining_weights))

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
			class_mode=self.config.get_class_mode(),
			shuffle=self.config.get_shuffle_training())

		val_generator = valid_datagen.flow_from_directory(
			self.config.get_validation_path(),
			target_size=(self.config.get_width(), self.config.get_height()),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode(),
			shuffle=self.config.get_shuffle_training())

		# steps for training
		steps = train_generator.n/train_generator.batch_size

		# store all training information while training the dataset
		self.history = History()

		print("Training network...")
		train_bottleneck_features = self.model.fit_generator(
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
			if self.config.get_model_name() != '' and self.config.get_should_save_model() and self.config.get_should_save_weights() and self.config.get_should_save_history() and self.config.get_should_save_bottleneck_features():
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

			if self.config.get_should_save_bottleneck_features() is True and self.config.get_model_name() != '':
				# Save features as numpy array
				np.save(open(self.config.get_model_output_path() + self.config.get_model_name() + '.features', 'w'), train_bottleneck_features)
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

		print("Inferencing folder(s)...")
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

	def create_project(self):
		def remove_folder(folder = None):
			if folder is None:
				return

			shutil.rmtree(folder)

		def create_folder(folder = None):
			if folder is None:
				return

			if not os.path.exists(folder):
				os.makedirs(folder)

		def generate_data(folder_path=None, output=None, percent=0.8):
			folders = [f for f in listdir(folder_path) if not isfile(join(folder_path, f))]

			for folder in folders:
				print(colored('Preparing '+folder_path+'', 'white') + " -> " + colored(output, 'green'))
				files = [f for f in listdir(folder_path + folder) if isfile(join(folder_path + folder, f))]

				total_files = len(files)
				total = round(total_files * percent)
				files = files[:total]

				for file in files:
					print(colored('Moving', 'blue') + " -> " + colored(file, 'red'))
					create_folder(output + folder + "/")
					src = folder_path + folder + "/" + file
					dest = output + folder + "/" + file
					shutil.move(src, dest)

		create_folder(self.config.get_project_train_path())
		create_folder(self.config.get_project_inference_path())
		create_folder(self.config.get_project_validation_path())
		create_folder(self.config.get_project_model_path())
		generate_data(self.config.get_project_dataset_path(), self.config.get_project_train_path(), 0.8)    # generate train data from dataset source with 80% files
		generate_data(self.config.get_project_dataset_path(), self.config.get_project_inference_path(), 1)  # generate inference data from dataset source, keep 100%
		generate_data(self.config.get_project_train_path(), self.config.get_project_validation_path(), 0.2) # generate validation data from train source with 20% files
		remove_folder(self.config.get_project_dataset_path())

		print("Project created successfully.")
		return self

	def plot_history(self):
		if self.history is None:
			return self

		print("Plotting history to images...")

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
	# Configurations for BlastML
	cfg = CFG(
			project={
				'project_name': 'shanynet.darknet',
				'root': '/ib/junk/junk/shany_ds/shany_proj/',
				'project_folder': 'final_project/',
				'dataset': 'dataset/',
				'train': 'train/',
				'inference': 'inference/',
				'validation': 'validation/',
				'model': 'model/',
			},
			image={
				'width': 224,
				'height': 224,
				'channels': 3
			},
			model={
				'load_model_embeddings': False,  # strip dropouts and fc layers
				'enable_saving': False,
				'save_model': True,
				'save_weights': True,
				'save_history': True,
				'save_bottleneck_features': True,
				'reset_learn_phase': True
			},
			hyper_params={
				'batch': 16,
				'epochs': 1,
				'classes': 5,
				'class_mode': 'sparse',
				'shuffle': False,
				'optimizer': 'adam',
				'loss_function': 'sparse_categorical_crossentropy',
				'compile_metrics': ['accuracy']
			},
			multithreading={
				'enable_multithreading': True,
				'threads': 5
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
			darknet={
				'cfg': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/model/yolov3.cfg',
				'weights': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/model/yolov3.weights',
				'taining_data': '/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/model/yolov3.weights',
				'enable_saving': True,
				'save_model': True,
				'save_weights': True
			})

	# Create a BlastML instance
	net = BlastML(cfg=cfg)

	# Create new project from dataset/ folder (contains only classes and their images)
	net.create_project()

	#  compile, train and evaluate a simple cnn instance
	# net.simple().compile().train().evaluate().infer()

	#  compile, train and evaluate a vgg16 instance
	#net.vgg16().compile().train().evaluate()

	#  compile, train and evaluate a resnet18 instance
	# net.resnet18().compile().train().evaluate()

	# Create, compile, train and evaluate a custom CNN instance
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

	# convert DarkNet model+weights to Keras model+weights
	# net.darknet_to_keras()

	# load model, create history (optional) and infer (test) your files (/inference)
	# cfg.threads = 1  # better to use 1 thread, but you can change it.
	# res = net.load_model()#.export_to_tf().plot_history().infer()
	# print(res)  # show embeddings/classification results

main()