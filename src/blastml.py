import os
import shutil
from PIL import Image
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import json

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img.load()
		img = img.convert('RGB')
	return img


class BlastML:
	def __init__(self, cfg=None):
		self.config = cfg

		if cfg is None:
			self.config = CFG()

		self.model = None
		self.predictions = None
		self.classes = None
		self.history = None

	def vgg16(self):
		# layer 1
		self.create()\
		.add_zero_padding(input_shape=(224, 224, 3))\
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
		model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))  # 32 filters, size of 3x3
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
		model.add(Dense(size=self.config.get_num_classes(), activation='softmax', name="layer_classes"))  # 100 classes

		model.summary()
		self.model = model

		return self

	def create(self):
		self.model = None
		self.model = Sequential()
		return self

	def add_2d(self, filters=32, kernel=(3,3), **kwargs):
		# stride (1,1)
		self.model.add(Conv2D(filters, kernel, **kwargs))  # 32 filters, size of 3x3
		return self

	def add_zero_padding(self, padding=(1, 1), **kwargs):
		self.model.add(ZeroPadding2D(padding=padding, **kwargs))
		return self

	def add_max_pooling(self, size=(2, 2), strides=None):
		self.model.add(MaxPooling2D(pool_size=size, strides=strides))  # 2x2 max pooling
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
					if name == 'layer_fc':
						del layer['config']
						del layer['class_name']
			while {} in layers:
				layers.remove({})

			model_json = json.dumps(json_file)
		else:
			json_file = open(self.config.get_model_output_path() + self.config.get_model_name() + '.json', 'r')
			model_json = json_file.read()

		model = model_from_json(model_json)
		model.load_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".h5", by_name=True)

		self.model = model

		# load history
		with open(self.config.get_model_output_path() + self.config.get_model_name() + '.history', 'rb') as f:
			self.history = pickle.load(f)

		return self

	def show_model_summary(self):
		self.model.summary()
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

		train_datagen = ImageDataGenerator(rescale=1./255)
		valid_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

		train_generator = train_datagen.flow_from_directory(
			self.config.get_train_path(),
			target_size=(224, 224),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode())

		val_generator = valid_datagen.flow_from_directory(
			self.config.get_validation_path(),
			target_size=(224, 224),
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
			target_size=(224, 224),
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
			target_size=(224, 224),
			batch_size=self.config.get_batch_size(),
			class_mode=self.config.get_class_mode())

		infer_datagen = ImageDataGenerator(rescale=1./255)
		infer_generator = infer_datagen.flow_from_directory(
			directory=self.config.get_infer_path(),
			target_size=(224, 224),
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
			predicted_class_indices=np.argmax(embeddings, axis=1)  # get index of classes
			labels = (train_generator.class_indices)
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

class CFG:
	def __init__(self, batch=10,
			epochs=100,
			enable_multithreading=True,
			threads=10,
			train_path='',
			classes=100,
			validation_path='',
			infer_path='',
			model_output_path='',
			model_name='',
			optimizer='sgd',
			class_mode='binary',
			loss_function='sparse_categorical_crossentropy',
			compile_metrics=['metrics'],
			enable_saving=False,
			save_model=False,
			save_weights=False,
			save_history=False,
			load_model_embeddings = False):
		self.batch = batch
		self.epochs = epochs
		self.enable_multithreading = enable_multithreading
		self.threads = threads
		self.train_path = train_path
		self.validation_path = validation_path
		self.infer_path = infer_path
		self.model_output_path = model_output_path
		self.model_name = model_name
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.compile_metrics = compile_metrics
		self.save_model = save_model
		self.save_weights = save_weights
		self.save_history = save_history
		self.enable_saving = enable_saving
		self.load_model_embeddings = load_model_embeddings
		self.num_classes = classes
		self.class_mode = class_mode

	def set_optimizer(self, optimizer=None):
		if optimizer is None:
			return self

		self.optimizer = optimizer

		return self

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


def main():
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

	cfg = CFG(batch=16,
			epochs=10,
			classes=5,
			enable_multithreading=True,
			threads=5,
			class_mode='sparse',
			optimizer='adam',
			loss_function='sparse_categorical_crossentropy',
			train_path='/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/train',
			validation_path='/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/validation',
			infer_path='/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/inference',
			model_output_path='/ib/junk/junk/shany_ds/shany_proj/dataset_final_project/model',
			model_name='shanynet',
			load_model_embeddings=False,  # strip dropouts and fc layers
			compile_metrics=['accuracy'],
			enable_saving=True,
			save_model=True,
			save_weights=True,
			save_history=True)

	# Create a BlastML instance
	Net = BlastML(cfg=cfg)

	# Create a very simple CNN instance
	# Net.simple().compile().train().evaluate().infer()

	# create a vgg16 instance
	Net.vgg16().compile().train().evaluate()

	# Create a custom CNN instance
	# Net.create()\
	# 	.add_2d(filters=32, kernel=(3, 3), activation="relu", padding='same', input_shape=(224, 224, 3))\
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
	# cfg.threads = 1  # better to use 1 thread, but you can change it.
	# res = Net.load_model().plot_history().infer()
	# print(res)  # show embeddings/classification results

main()