from BlastML.blastml import CFG, BlastML

# /Users/i337936/Desktop/
# /home/ubuntu/projects/

def main():
	# Configurations for BlastML
	cfg = CFG(
		project={
			'project_name': 'shanynet',
			'root': '/Users/i337936/Desktop/',
			'project_folder': 'final_project/',
			'dataset': 'dataset/',
			'train': 'train/',
			'inference': 'inference/',
			'validation': 'validation/',
			'model': 'model/',
		},
		image={
			'width': 128,
			'height': 128,
			'channels': 3
		},
		model={
			'load_model_embeddings': False,  # strip dropouts and fc layers
			'enable_saving': True,
			'save_model': True,
			'save_weights': True,
			'save_history': True,
			'save_bottleneck_features': True,
			'reset_learn_phase': True
		},
		hyper_params={
			'batch': 16,
			'epochs': 3000,
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
		object_detection={
			'yolo': {
				'cfg': 'model/darknet/yolov3-tiny-2c.cfg',
				'weights': 'model/darknet/yolov3-tiny-2c_final.weights',
				'training_data': 'model/darknet/data/train.txt',
				'class_names': 'model/darknet/data/classes.txt',
				'anchors': 'model/darknet/data/anchors.txt',
				'log': 'model/darknet/data/log',
				'rectlabel_csv': 'model/darknet/data/annotations.csv',
				'bboxes_font': 'model/darknet/data/Arial.ttf',
				"score": 0.3,
				"iou": 0.45,
				"model_image_size": (416, 416),
				"gpu_num": 1,
				'enable_saving': True,
				'save_model': True,
				'save_weights': True,
				'clusters': 6,
				'auto_estimate_anchors': True,
				'draw_bboxes': True,
				'exclude_infer_classes': ['guns', 'humans', 'knifes'],
				'enable_transfer_learning': True,
				'transfer_learning_epoch_ratio': [300, 150]
			}
		},
		gan={
			'dcgan': {
				"save_images_interval": 10,
				"random_noise_dimension": 100,
				"optimizer": {
					"type": 'adam',
					"learning_rate": 0.0002,
					"beta_1": 0.5
				}
			},
			'srgan': {
				"train_test_ratio": 0.8,
				"downscale_factor": 4
			}
		})

	# Create a BlastML instance
	net = BlastML(cfg=cfg)

	##################
	#  CNN Examples  #
	##################

	# Create new project from dataset/ folder (contains only classes and their images)
	# net.create_project()

	#  compile, train and evaluate a simple cnn instance
	# net.simple().compile().train().evaluate().infer()

	#  compile, train and evaluate a vgg16 instances
	# net.vgg16().compile().train().evaluate()

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

	# Load model
	# net.load_model()

	# plot (save as images) history
	# net.plot_history()

	# Infer images
	# cfg.threads = 1  # better to use 1 thread, but you can change it.
	# res = net.infer()
	# print(res)  # show embeddings/classification results


	############################
	#   DCGAN (GAN) Examples   #
	############################

	net.gan().srgan().train()

	############################
	#   YOLO/Darknet Examples  #
	############################

	# convert DarkNet model+weights to Keras model+weights
	# net.yolo().export_to_keras()

	# Calculate YOLOv3 anchors (this is done automatically) and save them to anchors.txt (check config)
	# net.yolo().generate_anchors()

	# Convert RectLabel csv export file to YOLOv3 format used in this BlastML implementation
	# net.yolo().rectLabel_to_YOLOv3()

	# train yolo model using DarkNet with model/data
	# net.yolo().create().compile().train()

	# infer yolo model
	# net.yolo().load_model().infer()

	# convert yolo model to protobuf
	# net.yolo().load_model().export_to_pb()

	# infer yolo model with webcame
	# net.yolo().load_model().infer_webcam()

main()