# BlastML
BlastML is a Lightning Fast Machine Learning Prototyping Library

With BlastML, you can prototype CNN networks (NLP to be added in the future) with ease.
BlastML uses Keras (TFlow) as an underline library, but it makes it much more easier than Keras itself.

### Projects
BlastML let you create a new project which aims to help unite 
the entire Machine Learning into single folder structure.

#### How to create a new project:
_Note_: before doing any training/inferencing on our data, it's best to make sure 
we collect data before hand. BlastML uses this dataset format:
```
root/
----project/
---------dataset/
------------Class1/
----------------File1
----------------File2
----------------File3
----------------<...>
------------Class2/
----------------File1
----------------File2
----------------File3
----------------<...>
------------<...>
```
Once you setup your dataset in that format, and you specified the 
correct paths in the project configuration settings, you just call 
<strong>create_project()</strong> method inside your net instance.<br/><br/>
Bt default, BlastML uses the standard 80:20 train, inference, validation file distribution.
Once the data has been processed, you can easily train, validate, infer your data with your 
CNN of your choice.

### CNN (Convolutional Neural Networks) 
BlastML contains implementations of various well known and tested CNN such as:
1. Simple (Basic CNN)
2. VGG-16
3. ResNet-18

### CNN builder
BlastML contains easy CNN building blocks which allows you to focus on building and testing your own CNN 
without spending too much time on function names, parameters and other mixed setups.
To build a custom network you simple call the create() function.

Custom CNN Example:
```
Net.create()
.add_2d(filters=32, kernel=(3, 3), activation="relu", padding='same', input_shape=(224, 224, 3))
.add_2d(filters=32, kernel=(3, 3), activation='relu')
.add_max_pooling()
.add_dropout()
.add_basic_block()
.add_basic_block()
.add_flatten()
.add_dense(size=512, activation='relu', name="layer_features")
.add_dense(size=cfg.get_num_classes(), activation='softmax', name="layer_classes")
.show_model_summary()
.compile()
.train()
.evaluate()
```
	
### How to use BlastML

##### Train and Evaluate
BlastML makes it so easy to use, that all you need is to:
1. Create data-set folders with data: 
    1. create a /train folder
    2. create a /validation folder
    3. create an /inference folder
    4. create a /model folder (where the model will be saved)
2. Create your BlastML configuration (CFG):
    1. set your train/validation/inference/model folders.
    2. enter the number of epochs.
    3. enter number of batches (images to process each batch).
    4. enter number of classes.
    5. enter number of thread (if multithreading is enabled).
    6. select optimizer (e.g: adam, sgd, rmsprop ...etc.).
    7. select a 'loss function' (e.g: sparse_categorical_crossentropy, binary_crossentropy ...etc.).
    8. select model name
    9. class_mode: sparse, categorical, binary (type of classification)
    10. [optional] enable global saving mode:
        1. save model
        2. save weights
        3. save history
    11. select your augmentation params
3. Create a BlastML(CFG) instance (attach configuration to BlastML)
4. Attach a CNN to BlastML.
5. Compile the CNN.
6. Train (get accuracy+loss for your model + validation accuracy+loss).
7. Evaluate (see how your model evaluated against never seen data).

Training is easy:
```
net = BlastML(cfg=cfg)
net.vgg16().compile().train().evaluate()
```

##### Inferencing and Plotting Results
1. set number of threads to 1 (makes sure threads don't overlap your data-set)
2. load model from disk to memory
3. [optional] plot history to /model folder (creates 2 images for loss and accuracy)
4. infer the data-set (get embeddings/classification results)


### Object Detection
BlastML supports DarkNet (YOLO v3) for Object Detections.<br/>
see: https://github.com/qqwweee/

<strong>Features:</strong>
1. Converting DarkNet weights to Keras weights (h5 file)
2. Training DarkNet with our own dataset (using YOLO v3 pre-trained and converted weights to h5)
    1. create model/darknet/data
        1. put anchors.txt
        1. put classes.txt
        1. put train.txt
    2. create model/darknet/data/log/ (for TensorBoard logs results)

_Note_: the training part is done in 2 stages:
1. 1-50 epochs with freezed 42/44 layers
2. 51-100 with un-freezed layers.

I have uploaded a demo (zipped) darknet/ folder, get it here:
[here](https://www.dropbox.com/s/a9l2nxsubq601wg/darknet.zip?dl=1)

Training is easy:
```
net = BlastML(cfg=cfg)
net.darknet().create().compile().train()
```

After training is completed, 2 .h5 (weights) file will be created
 1. 1-50 staged .h5 <**project name>.darknet.trained.stage.h5**)
 2. 51-100 trained .h5 (ie: <**project name>.darknet.trained.h5**) 

Next we can inference our videos/images.

#### Other Features
1. BlastML can Export Keras model to TensorFlow graph