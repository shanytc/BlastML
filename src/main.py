from BlastML import CFG, BlastML

def main():
	# Configurations for BlastML
	cfg = CFG(
		project={
			'project_name': 'shanynet',
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
			'enable_saving': True,
			'save_model': True,
			'save_weights': True,
			'save_history': True,
			'save_bottleneck_features': True,
			'reset_learn_phase': True
		},
		hyper_params={
			'batch': 32,
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
		object_detection={
			'yolo':{
				'cfg': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/yolov3-1c.cfg',
				'weights': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/yolov3.weights',
				'training_data': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/data/train.txt',
				'class_names': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/data/classes.txt',
				'anchors': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/data/anchors.txt',
				'log': '/ib/junk/junk/shany_ds/shany_proj/final_project/model/darknet/data/log',
				"score": 0.3,
				"iou": 0.45,
				"model_image_size": (416, 416),
				"gpu_num": 1,
				'enable_saving': True,
				'save_model': True,
				'save_weights': True
			}
		})

	# Create a BlastML instance
	net = BlastML(cfg=cfg)

	# Create new project from dataset/ folder (contains only classes and their images)
	# net.create_project()

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
	# net.yolo().export_to_keras()

	# train yolo model using darknet with model/data
	# net.yolo().create().compile().train()
	net.yolo().load_model().infer()

# load model, create history (optional) and infer (test) your files (/inference)
# cfg.threads = 1  # better to use 1 thread, but you can change it.
# res = net.load_model()#.export_to_tf().plot_history().infer()
# print(res)  # show embeddings/classification results

main()