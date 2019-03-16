# BlastML
BlastML is a Lightning Fast Machine Learning Prototyping Library

With BlastML, you can prototype CNN networks (NLP to be added in the future) with ease.
BlastML uses Keras (TFlow) as an underline library, but it makes it much more easier than Keras itself.

### CNN 
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
3. create a BlastML(CFG) instance (attach configuration to BlastML)
4. attach a CNN to BlastML.
5. compile the CNN.
6. train (get accuracy+loss for your model + validation accuracy+loss).
7. evaluate (see how your model evaluated against never seen data).

##### Inferencing and Plotting Results
1. set number of threads to 1 (makes sure threads don't overlap your data-set)
2. load model from disk to memory
3. [optional] plot history to /model folder (creates 2 images for loss and accuracy)
4. infer the data-set (get embeddings/classification results)