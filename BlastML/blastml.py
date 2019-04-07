import os
from os import listdir, makedirs
from os.path import isfile, join
import io
from termcolor import colored
import shutil
import configparser
from collections import defaultdict
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential, model_from_json
from keras.layers import UpSampling2D, Concatenate, Add, Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, BatchNormalization, LeakyReLU, AveragePooling2D, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import History,TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import json
import requests
import pandas as pd
import colorsys
from urllib.request import urlopen
from timeit import default_timer as timer
from functools import reduce
from PIL import Image, ImageFont, ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

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
	def __init__(self, project={}, image={}, augmentation={}, hyper_params={}, multithreading={}, model={}, object_detection={}):
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
		self.darknet_enable_saving = object_detection['yolo']['enable_saving']
		self.darknet_cfg = self.project_folder_path + object_detection['yolo']['cfg']
		self.darknet_weights = self.project_folder_path + object_detection['yolo']['weights']
		self.darknet_save_model = object_detection['yolo']['save_model']
		self.darknet_save_weight = object_detection['yolo']['save_weights']
		self.darknet_training_data = self.project_folder_path + object_detection['yolo']['training_data']
		self.darknet_classes_data = self.project_folder_path + object_detection['yolo']['class_names']
		self.darknet_anchors_data = self.project_folder_path + object_detection['yolo']['anchors']
		self.darknet_log_folder = self.project_folder_path + object_detection['yolo']['log']
		self.darknet_infer_score = object_detection['yolo']['score']
		self.darknet_infer_iou = object_detection['yolo']['iou']
		self.darknet_input_size = object_detection['yolo']['model_image_size']
		self.darknet_infer_gpu = object_detection['yolo']['gpu_num']
		self.darknet_clusters = object_detection['yolo']['clusters']
		self.darknet_rectlabel_csv = self.project_folder_path + object_detection['yolo']['rectlabel_csv']

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
		return self.darknet_training_data

	def get_darknet_classes(self):
		return self.darknet_classes_data

	def get_darknet_anchors(self):
		return self.darknet_anchors_data

	def get_darknet_log_folder(self):
		return self.darknet_log_folder

	def get_darknet_clusters(self):
		return self.darknet_clusters

	def get_darknet_rectlabel_csv(self):
		return self.darknet_rectlabel_csv

class DarkNet:
	def __init__(self, cfg=None):
		self.config = cfg
		self.annotation_path = self.config.get_darknet_training()
		self.log_dir = self.config.get_darknet_log_folder()
		self.classes_path = self.config.get_darknet_classes()
		self.anchors_path = self.config.get_darknet_anchors()
		self.class_names = self.get_classes(self.classes_path)
		self.num_classes = len(self.class_names)

		self.boxes = None
		self.scores = None
		self.classes = None
		self.infer_anchors = []
		self.infer_class_names = []
		self.colors = []
		self.session = None
		self.input_image_shape = None
		self.model = None
		self.anchors = None

	def DarknetConv2D(self, *args, **kwargs):
		# Wrapper to set Darknet parameters for Convolution2D.
		darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
		darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
		darknet_conv_kwargs.update(kwargs)
		return Conv2D(*args, **darknet_conv_kwargs)

	def DarknetConv2D_BN_Leaky(self, *args, **kwargs):
		# Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
		no_bias_kwargs = {'use_bias': False}
		no_bias_kwargs.update(kwargs)
		return self.compose(
			self.DarknetConv2D(*args, **no_bias_kwargs),
			BatchNormalization(),
			LeakyReLU(alpha=0.1))

	def resblock_body(self, x, num_filters, num_blocks):
		# A series of res-blocks starting with a downsampling Convolution2D
		# Darknet uses left and top padding instead of 'same' mode

		x = ZeroPadding2D(((1, 0),(1, 0)))(x)
		x = self.DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
		for i in range(num_blocks):
			y = self.compose(
				self.DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
				self.DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
			x = Add()([x,y])
		return x

	def darknet_body(self, x):
		# DarkNet body having 52 Convolution2D layers
		x = self.DarknetConv2D_BN_Leaky(32, (3, 3))(x)
		x = self.resblock_body(x, 64, 1)
		x = self.resblock_body(x, 128, 2)
		x = self.resblock_body(x, 256, 8)
		x = self.resblock_body(x, 512, 8)
		x = self.resblock_body(x, 1024, 4)

		return x

	def make_last_layers(self, x, num_filters, out_filters):
		# 6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
		x = self.compose(
			self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
			self.DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
			self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
			self.DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
			self.DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

		y = self.compose(
			self.DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
			self.DarknetConv2D(out_filters, (1, 1)))(x)

		return x, y

	def yolo_body(self, inputs, num_anchors, num_classes):
		# Create YOLOv3 model CNN body in Keras.
		darknet = Model(inputs, self.darknet_body(inputs))
		x, y1 = self.make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

		x = self.compose(
			self.DarknetConv2D_BN_Leaky(256, (1, 1)),
			UpSampling2D(2))(x)

		x = Concatenate()([x,darknet.layers[152].output])
		x, y2 = self.make_last_layers(x, 256, num_anchors*(num_classes+5))

		x = self.compose(
			self.DarknetConv2D_BN_Leaky(128, (1, 1)),
			UpSampling2D(2))(x)
		x = Concatenate()([x, darknet.layers[92].output])
		x, y3 = self.make_last_layers(x, 128, num_anchors*(num_classes+5))

		return Model(inputs, [y1, y2, y3])

	def tiny_yolo_body(self, inputs, num_anchors, num_classes):
		# Create Tiny YOLO_v3 model CNN body in keras.
		x1 = self.compose(
			self.DarknetConv2D_BN_Leaky(16, (3, 3)),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
			self.DarknetConv2D_BN_Leaky(32, (3, 3)),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
			self.DarknetConv2D_BN_Leaky(64, (3, 3)),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
			self.DarknetConv2D_BN_Leaky(128, (3, 3)),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
			self.DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)

		x2 = self.compose(
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
			self.DarknetConv2D_BN_Leaky(512, (3, 3)),
			MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
			self.DarknetConv2D_BN_Leaky(1024, (3, 3)),
			self.DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)

		y1 = self.compose(
			self.DarknetConv2D_BN_Leaky(512, (3, 3)),
			self.DarknetConv2D(num_anchors*(num_classes+5), (1, 1)))(x2)

		x2 = self.compose(
			self.DarknetConv2D_BN_Leaky(128, (1, 1)),
			UpSampling2D(2))(x2)

		y2 = self.compose(
			Concatenate(),
			self.DarknetConv2D_BN_Leaky(256, (3, 3)),
			self.DarknetConv2D(num_anchors*(num_classes+5), (1, 1)))([x2, x1])

		return Model(inputs, [y1, y2])

	def yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):
		# Convert final layer features to bounding box parameters.
		num_anchors = len(anchors)

		# Reshape to batch, height, width, num_anchors, box_params.
		anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

		grid_shape = K.shape(feats)[1:3]  # height, width
		grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
		grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
		grid = K.concatenate([grid_x, grid_y])
		grid = K.cast(grid, K.dtype(feats))

		feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

		# Adjust predictions to each spatial grid point and anchor size.
		box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
		box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
		box_confidence = K.sigmoid(feats[..., 4:5])
		box_class_probs = K.sigmoid(feats[..., 5:])

		if calc_loss:
			return grid, feats, box_xy, box_wh

		return box_xy, box_wh, box_confidence, box_class_probs

	def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
		# Get corrected boxes
		box_yx = box_xy[..., ::-1]
		box_hw = box_wh[..., ::-1]
		input_shape = K.cast(input_shape, K.dtype(box_yx))
		image_shape = K.cast(image_shape, K.dtype(box_yx))
		new_shape = K.round(image_shape * K.min(input_shape/image_shape))
		offset = (input_shape-new_shape)/2./input_shape
		scale = input_shape/new_shape
		box_yx = (box_yx - offset) * scale
		box_hw *= scale

		box_mins = box_yx - (box_hw / 2.)
		box_maxes = box_yx + (box_hw / 2.)
		boxes = K.concatenate([
			box_mins[..., 0:1],  # y_min
			box_mins[..., 1:2],  # x_min
			box_maxes[..., 0:1],  # y_max
			box_maxes[..., 1:2]  # x_max
		])

		# Scale boxes back to original image shape.
		boxes *= K.concatenate([image_shape, image_shape])
		return boxes

	def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
		# Process Conv layer output
		box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape)
		boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
		boxes = K.reshape(boxes, [-1, 4])
		box_scores = box_confidence * box_class_probs
		box_scores = K.reshape(box_scores, [-1, num_classes])
		return boxes, box_scores

	def yolo_eval(self, yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
		# Evaluate YOLO model on given input and return filtered boxes.
		num_layers = len(yolo_outputs)
		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
		input_shape = K.shape(yolo_outputs[0])[1:3] * 32
		boxes = []
		box_scores = []
		for l in range(num_layers):
			_boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
			boxes.append(_boxes)
			box_scores.append(_box_scores)
		boxes = K.concatenate(boxes, axis=0)
		box_scores = K.concatenate(box_scores, axis=0)

		mask = box_scores >= score_threshold
		max_boxes_tensor = K.constant(max_boxes, dtype='int32')
		boxes_ = []
		scores_ = []
		classes_ = []
		for c in range(num_classes):
			# TODO: use keras backend instead of tf.
			class_boxes = tf.boolean_mask(boxes, mask[:, c])
			class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
			nms_index = tf.image.non_max_suppression(
				class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
			class_boxes = K.gather(class_boxes, nms_index)
			class_box_scores = K.gather(class_box_scores, nms_index)
			classes = K.ones_like(class_box_scores, 'int32') * c
			boxes_.append(class_boxes)
			scores_.append(class_box_scores)
			classes_.append(classes)
		boxes_ = K.concatenate(boxes_, axis=0)
		scores_ = K.concatenate(scores_, axis=0)
		classes_ = K.concatenate(classes_, axis=0)

		return boxes_, scores_, classes_

	def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
		# Preprocess true boxes to training input format
		#
		# Parameters
		# ----------
		# true_boxes: array, shape=(m, T, 5)
		#	Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
		# input_shape: array-like, hw, multiples of 32
		# anchors: array, shape=(N, 2), wh
		# num_classes: integer
		#
		# Returns
		# -------
		# y_true: list of array, shape like yolo_outputs, xywh are reletive value

		assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
		num_layers = len(anchors)//3  # default setting
		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

		true_boxes = np.array(true_boxes, dtype='float32')
		input_shape = np.array(input_shape, dtype='int32')
		boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
		boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
		true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
		true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

		m = true_boxes.shape[0]
		grid_shapes = [input_shape//{0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
		y_true = [np.zeros((m,grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes), dtype='float32') for l in range(num_layers)]

		# Expand dim to apply broadcasting.
		anchors = np.expand_dims(anchors, 0)
		anchor_maxes = anchors / 2.
		anchor_mins = -anchor_maxes
		valid_mask = boxes_wh[..., 0]>0

		for b in range(m):
			# Discard zero rows.
			wh = boxes_wh[b, valid_mask[b]]
			if len(wh) == 0: continue
			# Expand dim to apply broadcasting.
			wh = np.expand_dims(wh, -2)
			box_maxes = wh / 2.
			box_mins = -box_maxes

			intersect_mins = np.maximum(box_mins, anchor_mins)
			intersect_maxes = np.minimum(box_maxes, anchor_maxes)
			intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
			intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
			box_area = wh[..., 0] * wh[..., 1]
			anchor_area = anchors[..., 0] * anchors[..., 1]
			iou = intersect_area / (box_area + anchor_area - intersect_area)

			# Find best anchor for each true box
			best_anchor = np.argmax(iou, axis=-1)

			for t, n in enumerate(best_anchor):
				for l in range(num_layers):
					if n in anchor_mask[l]:
						i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
						j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
						k = anchor_mask[l].index(n)
						c = true_boxes[b,t, 4].astype('int32')
						y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
						y_true[l][b, j, i, k, 4] = 1
						y_true[l][b, j, i, k, 5+c] = 1

		return y_true

	def box_iou(self, b1, b2):
		# Return iou tensor
		#
		# Parameters
		# ----------
		# b1: tensor, shape=(i1,...,iN, 4), xywh
		# b2: tensor, shape=(j, 4), xywh
		#
		# Returns
		# -------
		# iou: tensor, shape=(i1,...,iN, j)

		# Expand dim to apply broadcasting.
		b1 = K.expand_dims(b1, -2)
		b1_xy = b1[..., :2]
		b1_wh = b1[..., 2:4]
		b1_wh_half = b1_wh/2.
		b1_mins = b1_xy - b1_wh_half
		b1_maxes = b1_xy + b1_wh_half

		# Expand dim to apply broadcasting.
		b2 = K.expand_dims(b2, 0)
		b2_xy = b2[..., :2]
		b2_wh = b2[..., 2:4]
		b2_wh_half = b2_wh/2.
		b2_mins = b2_xy - b2_wh_half
		b2_maxes = b2_xy + b2_wh_half

		intersect_mins = K.maximum(b1_mins, b2_mins)
		intersect_maxes = K.minimum(b1_maxes, b2_maxes)
		intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
		intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
		b1_area = b1_wh[..., 0] * b1_wh[..., 1]
		b2_area = b2_wh[..., 0] * b2_wh[..., 1]
		iou = intersect_area / (b1_area + b2_area - intersect_area)

		return iou

	def yolo_loss(self, args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
		# Return yolo_loss tensor
		# Parameters
		# ----------
		# yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
		# y_true: list of array, the output of preprocess_true_boxes
		# anchors: array, shape=(N, 2), wh
		# num_classes: integer
		# ignore_thresh: float, the iou threshold whether to ignore object confidence loss
		# Returns
		# -------
		# loss: tensor, shape=(1,)

		num_layers = len(anchors)//3  # default setting
		yolo_outputs = args[:num_layers]
		y_true = args[num_layers:]
		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
		input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
		grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
		loss = 0
		m = K.shape(yolo_outputs[0])[0] # batch size, tensor
		mf = K.cast(m, K.dtype(yolo_outputs[0]))

		for l in range(num_layers):
			object_mask = y_true[l][..., 4:5]
			true_class_probs = y_true[l][..., 5:]

			grid, raw_pred, pred_xy, pred_wh = self.yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
			pred_box = K.concatenate([pred_xy, pred_wh])

			# Darknet raw box to calculate loss.
			raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
			raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
			raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
			box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

			# Find ignore mask, iterate over each of batch.
			ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
			object_mask_bool = K.cast(object_mask, 'bool')

			def loop_body(b, ignore_mask):
				true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b, ..., 0])
				iou = self.box_iou(pred_box[b], true_box)
				best_iou = K.max(iou, axis=-1)
				ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
				return b+1, ignore_mask

			_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
			ignore_mask = ignore_mask.stack()
			ignore_mask = K.expand_dims(ignore_mask, -1)

			# K.binary_crossentropy is helpful to avoid exp overflow.
			xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
			wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[..., 2:4])
			confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
			class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

			xy_loss = K.sum(xy_loss) / mf
			wh_loss = K.sum(wh_loss) / mf
			confidence_loss = K.sum(confidence_loss) / mf
			class_loss = K.sum(class_loss) / mf
			loss += xy_loss + wh_loss + confidence_loss + class_loss
			if print_loss:
				loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='yolo loss: ')
		return loss

	def compose(self, *funcs):
		# Compose arbitrarily many functions, evaluated left to right.
		# Reference: https://mathieularose.com/function-composition-in-python/

		# return lambda x: reduce(lambda v, f: f(v), funcs, x)
		if funcs:
			return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
		else:
			raise ValueError('Composition of empty sequence not supported.')

	def letterbox_image(self, image, size):
		# resize image with unchanged aspect ratio using padding
		iw, ih = image.size
		w, h = size
		scale = min(w/iw, h/ih)
		nw = int(iw*scale)
		nh = int(ih*scale)

		image = image.resize((nw, nh), Image.BICUBIC)
		new_image = Image.new('RGB', size, (128, 128, 128))
		new_image.paste(image, ((w-nw)//2, (h-nh)//2))
		return new_image

	def rand(self, a=0, b=1):
		return np.random.rand()*(b-a) + a

	def get_random_data(self, annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
		# random pre-processing for real-time data augmentation
		line = annotation_line.split()
		image = Image.open(line[0])
		iw, ih = image.size
		h, w = input_shape
		box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

		if not random:
			# resize image
			scale = min(w/iw, h/ih)
			nw = int(iw*scale)
			nh = int(ih*scale)
			dx = (w-nw)//2
			dy = (h-nh)//2
			image_data = 0
			if proc_img:
				image = image.resize((nw,nh), Image.BICUBIC)
				new_image = Image.new('RGB', (w, h), (128, 128, 128))
				new_image.paste(image, (dx, dy))
				image_data = np.array(new_image)/255.

			# correct boxes
			box_data = np.zeros((max_boxes, 5))
			if len(box) > 0:
				np.random.shuffle(box)
				if len(box) > max_boxes:
					box = box[:max_boxes]
				box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
				box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
				box_data[:len(box)] = box

			return image_data, box_data

		# resize image
		new_ar = w/h * self.rand(1-jitter, 1+jitter)/self.rand(1-jitter, 1+jitter)
		scale = self.rand(.25, 2)
		if new_ar < 1:
			nh = int(scale*h)
			nw = int(nh*new_ar)
		else:
			nw = int(scale*w)
			nh = int(nw/new_ar)
		image = image.resize((nw, nh), Image.BICUBIC)

		# place image
		dx = int(self.rand(0, w-nw))
		dy = int(self.rand(0, h-nh))
		new_image = Image.new('RGB', (w,h), (128, 128, 128))
		new_image.paste(image, (dx, dy))
		image = new_image

		# flip image or not
		flip = self.rand() < .5
		if flip:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)

		# distort image
		hue = self.rand(-hue, hue)
		sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
		val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
		x = rgb_to_hsv(np.array(image)/255.)
		x[..., 0] += hue
		x[..., 0][x[..., 0] > 1] -= 1
		x[..., 0][x[..., 0] < 0] += 1
		x[..., 1] *= sat
		x[..., 2] *= val
		x[x > 1] = 1
		x[x < 0] = 0
		image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

		# correct boxes
		box_data = np.zeros((max_boxes,5))
		if len(box) > 0:
			np.random.shuffle(box)
			box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
			box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy

			if flip:
				box[:, [0, 2]] = w - box[:, [2, 0]]

			box[:, 0:2][box[:, 0:2] < 0] = 0
			box[:, 2][box[:, 2] > w] = w
			box[:, 3][box[:, 3] > h] = h
			box_w = box[:, 2] - box[:, 0]
			box_h = box[:, 3] - box[:, 1]
			box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

			if len(box) > max_boxes:
				box = box[:max_boxes]

			box_data[:len(box)] = box

		return image_data, box_data

	def get_classes(self, classes_path):
		# loads the classes
		with open(classes_path) as f:
			class_names = f.readlines()
		class_names = [c.strip() for c in class_names]
		return class_names

	def get_anchors(self, anchors_path):
		# loads the anchors from a file
		with open(anchors_path) as f:
			anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		return np.array(anchors).reshape(-1, 2)

	def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
		# data generator for fit_generator
		n = len(annotation_lines)
		i = 0
		while True:
			image_data = []
			box_data = []
			for b in range(batch_size):
				if i == 0:
					np.random.shuffle(annotation_lines)
				image, box = self.get_random_data(annotation_lines[i], input_shape, random=True)
				image_data.append(image)
				box_data.append(box)
				i = (i+1) % n
			image_data = np.array(image_data)
			box_data = np.array(box_data)
			y_true = self.preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
			yield [image_data, *y_true], np.zeros(batch_size)

	def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
		n = len(annotation_lines)

		if n == 0 or batch_size <= 0:
			return None

		return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

	def create_model(self, input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='model_data/yolo_weights.h5'):
		# create the training model

		K.clear_session()  # get a new session
		image_input = Input(shape=(None, None, 3))
		h, w = input_shape
		num_anchors = len(anchors)

		y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l], num_anchors//3, num_classes+5)) for l in range(3)]

		model_body = self.yolo_body(image_input, num_anchors//3, num_classes)
		print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

		if load_pretrained:
			model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
			print('Load weights {}.'.format(weights_path))
			if freeze_body in [1, 2]:
				# Freeze darknet53 body or freeze all but 3 output layers.
				num = (185, len(model_body.layers)-3)[freeze_body-1]
				for i in range(num):
					model_body.layers[i].trainable = False
				print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

		model_loss = Lambda(self.yolo_loss, output_shape=(1,), name='yolo_loss', arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
		model = Model([model_body.input, *y_true], model_loss)

		return model

	def create_tiny_model(self, input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path=''):
		# create the training model, for Tiny YOLOv3
		K.clear_session()  # get a new session
		image_input = Input(shape=(None, None, 3))
		h, w = input_shape
		num_anchors = len(anchors)

		y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l], num_anchors//2, num_classes+5)) for l in range(2)]

		model_body = self.tiny_yolo_body(image_input, num_anchors//2, num_classes)
		print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

		if load_pretrained:
			model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
			print('Load weights {}.'.format(weights_path))
			if freeze_body in [1, 2]:
				# Freeze the darknet body or freeze all but 2 output layers.
				num = (20, len(model_body.layers)-2)[freeze_body-1]
				for i in range(num):
					model_body.layers[i].trainable = False
				print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

		model_loss = Lambda(self.yolo_loss, output_shape=(1,), name='yolo_loss', arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})([*model_body.output, *y_true])
		model = Model([model_body.input, *y_true], model_loss)

		return model

	def get_training_files(self):
		training_files = []

		with open(self.config.get_darknet_training()) as f:
			lines = f.readlines()

		for l in lines:
			train_file = self.config.get_project_train_path() + l.split(' ')[0]
			val_file = self.config.get_project_validation_path() + l.split(' ')[0]

			if isfile(train_file) and self.config.get_project_train_path()+l not in training_files:
				training_files.append(self.config.get_project_train_path() + l.replace("\n", ""))
			else:
				if isfile(val_file) and self.config.get_project_validation_path()+l not in training_files:
					training_files.append(self.config.get_project_validation_path() + l.replace("\n", ""))

		return training_files

	def create(self):
		self.model = None
		input_shape = (416, 416)  # multiple of 32, hw

		# create anchors based on training+validation files before training
		self.generate_anchors()
		self.anchors = self.get_anchors(self.anchors_path)

		is_tiny_version = len(self.anchors) == 6  # default setting
		if is_tiny_version:
			self.model = self.create_tiny_model(input_shape, self.anchors, self.num_classes, freeze_body=2, weights_path=self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.h5")
		else:
			self.model = self.create_model(input_shape, self.anchors, self.num_classes, freeze_body=2, weights_path=self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.h5")  # make sure you know what you freeze

		return self

	def compile(self,learning_rate=1e-3):
		if self.model is None:
			return self

		# use custom yolo_loss Lambda layer.
		self.model.compile(optimizer=Adam(lr=learning_rate, amsgrad=True), metrics=self.config.get_compile_metrics(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

		return self

	def train(self):
		if self.model is None:
			return self

		input_shape = (416, 416)  # multiple of 32, hw

		logging = TensorBoard(log_dir=self.log_dir)
		checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)  # Reduce learning rate when a metric has stopped improving.
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)  # Early stop if validation loss is not improving

		val_split = 0.1
		lines = self.get_training_files()  # get training data (combination of train + validation)
		np.random.seed(10101)
		np.random.shuffle(lines)
		np.random.seed(None)
		num_val = int(len(lines)*val_split)
		num_train = len(lines) - num_val

		# Train with frozen layers first, to get a stable loss.
		# Adjust num epochs to your data set.
		# This step is enough to obtain a not bad model.

		batch_size = self.config.get_batch_size()
		print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

		self.model.fit_generator(self.data_generator_wrapper(lines[:num_train], batch_size, input_shape, self.anchors, self.num_classes),
			steps_per_epoch=max(1, num_train//batch_size),
			validation_data=self.data_generator_wrapper(lines[num_train:], batch_size, input_shape, self.anchors, self.num_classes),
			validation_steps=max(1, num_val//batch_size),
			epochs=self.config.get_num_epochs()//2,
			initial_epoch=0,
			callbacks=[logging, checkpoint])

		self.model.save_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.trained.stage.h5")

		# Unfreeze and continue training, to fine-tune.
		# Train longer if the result is not good.
		for i in range(len(self.model.layers)):
			self.model.layers[i].trainable = True

		# re-compile with smaller learning rate
		self.compile(1e-4)

		print('Unfreeze all of the layers.')

		batch_size = self.config.get_batch_size()  # note that more GPU memory is required after unfreezing the body
		print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

		self.model.fit_generator(self.data_generator_wrapper(lines[:num_train], batch_size, input_shape, self.anchors, self.num_classes),
			steps_per_epoch=max(1, num_train//batch_size),
			validation_data=self.data_generator_wrapper(lines[num_train:], batch_size, input_shape, self.anchors, self.num_classes),
			validation_steps=max(1, num_val//batch_size),
			epochs=self.config.get_num_epochs(),
			initial_epoch=self.config.get_num_epochs()//2,
			callbacks=[logging, checkpoint, reduce_lr, early_stopping])

		self.model.save_weights(self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.trained.h5")

		print('Training complete.')

		return self

	def load_model(self):
		# TODO: remove this guys, since we already defined them above
		def _get_class(path):
			with open(path) as f:
				class_names = f.readlines()
			class_names = [c.strip() for c in class_names]
			return class_names

		def _get_anchors(path):
			with open(path) as f:
				anchors = f.readline()
			anchors = [float(x) for x in anchors.split(',')]
			return np.array(anchors).reshape(-1, 2)

		self.infer_class_names = _get_class(self.classes_path)
		self.infer_anchors = _get_anchors(self.anchors_path)
		self.session = K.get_session()

		model_path = os.path.expanduser(self.config.get_model_output_path() + self.config.get_model_name() + ".darknet.trained.h5")
		assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

		# Load model, or construct model and load weights.
		num_anchors = len(self.infer_anchors)
		num_classes = len(self.infer_class_names)
		is_tiny_version = num_anchors == 6  # default setting

		try:
			self.model = load_model(model_path, compile=False)
		except:
			self.model = self.tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) if is_tiny_version else self.yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
			self.model.load_weights(model_path)  # make sure model, anchors and classes match
		else:
			assert self.model.layers[-1].output_shape[-1] == num_anchors/len(self.model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'

		print('{} model, anchors, and classes loaded.'.format(model_path))

		# Generate colors for drawing bounding boxes.
		hsv_tuples = [(x / len(self.infer_class_names), 1., 1.) for x in range(len(self.infer_class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

		np.random.seed(10101)  # Fixed seed for consistent colors across runs.
		np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		np.random.seed(None)  # Reset seed to default.

		# Generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2, ))

		if self.config.darknet_infer_gpu >= 2:
			self.model = multi_gpu_model(self.model, gpus=self.config.darknet_infer_gpu)

		self.boxes, self.scores, self.classes = self.yolo_eval(self.model.output, self.infer_anchors, len(self.infer_class_names), self.input_image_shape, score_threshold=self.config.darknet_infer_score, iou_threshold=self.config.darknet_infer_iou)

		return self

	def infer(self):
		# load image using Pillow
		def img_loader(path):
			with open(path, 'rb') as f:
				img = Image.open(f)
				img.load()
				img = img.convert('RGB')
			return img

		def img_loader_from_url(url):
			img = Image.open(urlopen(url))
			#img.load()
			#img = img.convert('RGB')
			return img

		# image_path = 'https://dsvf96nw4ftce.cloudfront.net/images/thumbnails/460/460/detailed/2/dog-bike-tow-leash-action2.jpg?t=1527115978'
		# image = img_loader_from_url(image_path)

		# try to find boxes in inference/ folder
		files = [f for f in listdir(self.config.get_infer_path() + 'terrorists/') if isfile(join(self.config.get_infer_path() + 'terrorists/', f))]
		for f in files:
			if f == ".DS_Store":
				continue

			image_path = self.config.get_infer_path() + 'terrorists/' + f
			image = img_loader(image_path)

			if self.config.darknet_input_size != (None, None):
				assert self.config.darknet_input_size[0] % 32 == 0, 'Multiples of 32 required'
				assert self.config.darknet_input_size[1] % 32 == 0, 'Multiples of 32 required'
				boxed_image = self.letterbox_image(image, tuple(reversed(self.config.darknet_input_size)))
			else:
				new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
				boxed_image = self.letterbox_image(image, new_image_size)
			image_data = np.array(boxed_image, dtype='float32')

			image_data /= 255.
			image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

			start = timer()

			out_boxes, out_scores, out_classes = self.session.run(
				[self.boxes, self.scores, self.classes],
				feed_dict={
					self.model.input: image_data,
					self.input_image_shape: [image.size[1], image.size[0]],
					K.learning_phase(): 0
				})

			#end = timer()

			# print(end - start)
			print("Boxes:" + str(len(out_boxes)) + " - " + f)

			font = ImageFont.truetype(font='Arial.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
			thickness = (image.size[0] + image.size[1]) // 300

			for i, c in reversed(list(enumerate(out_classes))):
				predicted_class = self.class_names[c]
				box = out_boxes[i]
				score = out_scores[i]

				label = '{} {:.2f}'.format(predicted_class, score)
				draw = ImageDraw.Draw(image)
				label_size = draw.textsize(label, font)

				top, left, bottom, right = box
				top = max(0, np.floor(top + 0.5).astype('int32'))
				left = max(0, np.floor(left + 0.5).astype('int32'))
				bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
				right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
				print(label, (left, top), (right, bottom))

				if top - label_size[1] >= 0:
					text_origin = np.array([left, top - label_size[1]])
				else:
					text_origin = np.array([left, top + 1])

				# My kingdom for a good redistributable image drawing library.
				for i in range(thickness):
					draw.rectangle(
						[left + i, top + i, right - i, bottom - i],
						outline=self.colors[c])
				draw.rectangle(
					[tuple(text_origin), tuple(text_origin + label_size)],
					fill=self.colors[c])
				draw.text(text_origin, label, fill=(0, 0, 0), font=font)
				del draw

				#image.save(image_path)

	def generate_anchors(self):
		def iou(boxes, clusters, cluster_number):  # 1 box -> k clusters
			n = boxes.shape[0]
			k = cluster_number

			box_area = boxes[:, 0] * boxes[:, 1]
			box_area = box_area.repeat(k)
			box_area = np.reshape(box_area, (n, k))

			cluster_area = clusters[:, 0] * clusters[:, 1]
			cluster_area = np.tile(cluster_area, [1, n])
			cluster_area = np.reshape(cluster_area, (n, k))

			box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
			cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
			min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

			box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
			cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
			min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
			inter_area = np.multiply(min_w_matrix, min_h_matrix)

			result = inter_area / (box_area + cluster_area - inter_area)

			return result

		def avg_iou(boxes, clusters, cluster_number):
			accuracy = np.mean([np.max(iou(boxes, clusters, cluster_number), axis=1)])
			return accuracy

		def kmeans(boxes, k, dist=np.median):
			box_number = boxes.shape[0]
			distances = np.empty((box_number, k))
			last_nearest = np.zeros((box_number,))
			np.random.seed()
			clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters
			while True:
				distances = 1 - iou(boxes, clusters, k)

				current_nearest = np.argmin(distances, axis=1)
				if (last_nearest == current_nearest).all():
					break  # clusters won't change
				for cluster in range(k):
					clusters[cluster] = dist(  # update clusters
						boxes[current_nearest == cluster], axis=0)

				last_nearest = current_nearest

			return clusters

		dataSet = []
		cluster_number = self.config.get_darknet_clusters()

		# get files to cxalculate anchors from
		files = self.get_training_files()  # get training data (combination of train + validation)

		# self.config.get_darknet_anchors()
		for line in files:
			infos = line.split(" ")
			length = len(infos)
			for i in range(1, length):
				width = int(infos[i].split(",")[2])  # - int(infos[i].split(",")[0])
				height = int(infos[i].split(",")[3])  # - int(infos[i].split(",")[1])
				dataSet.append([width, height])

		all_boxes = np.array(dataSet)
		result = kmeans(all_boxes, k=cluster_number)
		f = open(self.config.get_darknet_anchors(), 'w')
		row = np.shape(result)[0]
		for i in range(row):
			if i == 0:
				x_y = "%d,%d" % (result[i][0], result[i][1])
			else:
				x_y = ", %d,%d" % (result[i][0], result[i][1])
			f.write(x_y)
		f.close()

		print("K anchors:\n {}".format(result))
		print("Accuracy: {:.2f}%".format(avg_iou(all_boxes, result, cluster_number) * 100))

		return self

	def generate_anchors2(self):
		annotation_dims = []
		size = np.zeros((1, 1, 3))
		width_in_cfg_file = 416
		height_in_cfg_file = 416

		# get files to cxalculate anchors from
		lines = self.get_training_files()  # get training data (combination of train + validation)
		lines = [l.split(' ')[0] for l in lines]

		# self.config.get_darknet_anchors()
		#for line in lines:

		return self

	def rectLabel_to_YOLOv3(self):
		# for RectLabel tool, go here: https://github.com/ryouchinsa/Rectlabel-support

		# create labels index
		labels = {}
		with open(self.config.get_darknet_classes(), 'r') as f:
			lines = f.readlines()

		index = 0
		for c in lines:
			labels[c] = index
			index += 1

		# convert RectLabel tool (xml->csv) to YOLOv3 format
		file = self.config.get_darknet_rectlabel_csv()
		os.remove(self.config.get_darknet_training())
		f = open(self.config.get_darknet_training(), "w")
		data = pd.read_csv(file, sep='\t')

		for i, j in data.iterrows():
			cols = list(j)
			row_data = cols[0].split(",[")
			json_data = json.loads("["+row_data[1])
			file_name = json_data[0]['label'] + "/" + os.path.basename(row_data[0])

			file_data = file_name + " "
			for box in json_data:
				file_data += str(box['coordinates']['x']) + "," + str(box['coordinates']['y']) + "," + str(box['coordinates']['x'] + box['coordinates']['width']) + "," + str(box['coordinates']['y'] + box['coordinates']['height']) + "," + str(labels[box['label']]) + " "

			file_data = file_data.strip()
			f.write(file_data + "\r\n")

		f.close()

		print("Successfully exported RectLabel to YOLOv3 train format.")

		return self

	def export_to_keras(self):
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

class BlastML:
	def __init__(self, cfg=None):
		self.config = cfg

		# Use default configuration settings
		if cfg is None:
			self.config = CFG()

		self.model = None
		self.predictions = None
		self.classes = None
		self.history = None
		self.bottleneck = None
		self.Darknet = DarkNet(self.config)

	# DarkNet CNN
	def yolo(self):
		return self.Darknet

	# ResNet18 CNN
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

	# VGG16 CNN
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

	# Simple CNN
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
		def remove_folder(folder=None):
			if folder is None:
				return

			shutil.rmtree(folder)

		def create_folder(folder=None):
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
