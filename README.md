#Deep Neural Networks with GPU support

This is a Java implementation of some of the algorithms for training deep neural networks. GPU support is provided via the OpenCL and Aparapi.
The architecture is designed with modularity, extensibility and pluggability in mind.

####Git structure
I'm using the [git-flow](https://github.com/nvie/gitflow) model. The most stable (but older) sources are available in the [_master_](https://github.com/ivan-vasilev/neuralnetworks/tree/master) branch, while the latest ones are in the [_develop_](https://github.com/ivan-vasilev/neuralnetworks/tree/develop) branch.

**If you want to use the previous Java 7 compatible version you can check out [this](https://github.com/ivan-vasilev/neuralnetworks/releases/tag/v0.1.0-alpha) release.**

##Neural network types
* Multilayer perceptron
* Restricted Boltzmann Machine
* Autoencoder
* Deep belief network
* Stacked autoencodeer
* Convolutional networks with max pooling, average poolng and [stochastic pooling](http://techtalks.tv/talks/stochastic-pooling-for-regularization-of-deep-convolutional-neural-networks/58106/).
* Maxout networks (work-in-progress)

##Training algorithms
* Backpropagation - supports multilayer perceptrons, convolutional networks and [dropout](http://arxiv.org/pdf/1207.0580.pdf).
* Contrastive divergence and persistent contrastive divergence implemented using [these](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239) and [these](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) guidelines.
* Greedy layer-wise training for deep networks - works for stacked autoencoders and DBNs, but supports any kind of training.

All the algorithms support GPU execution. 

Out of the box supported datasets are [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10/CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) (experimental), [IRIS](http://archive.ics.uci.edu/ml/datasets/Iris) and XOR, but you can easily implement your own.

Experimental support of RGB image preprocessing operations - affine transformations, cropping, and color scaling (see Generaltest.java -> testImageInputProvider).

##Activation functions
* Logistic
* Tanh
* Rectifiers
* Softplus
* Softmax
* Weighted sum

All the functions support GPU execution. They can be applied to all types of networks and all training algorithms. You can also implement new activations.

##How to build the library
* **Java 8**.
* To build the project you need [gradle](http://www.gradle.org/) or [maven](http://maven.apache.org/). If you don't use any of these you can go to the project folder and execute the _gradlew_ console command, which will automatically setup gradle environment for you.
* I'm also uploading the latest jar file (with bundled dependencies and sources) [here](https://github.com/ivan-vasilev/neuralnetworks/tree/master/build/libs).
* Depending on your environment you might need to download the relevant aparapi .dll or .so file (located in the root of each archive) from [here](https://code.google.com/p/aparapi/downloads/list) and add it's location to the system PATH variable. (This)[https://code.google.com/p/aparapi/wiki/DevelopersGuideLinux] is a guide on how to set up OpenCL in linux environment.

##How to run the samples
The samples are organized as unit tests. If you want see examples on various popular datasets you can go to [nn-samples/src/test/java/com/github/neuralnetworks/samples/](https://github.com/ivan-vasilev/neuralnetworks/tree/9e569aa7c9a4d724cf3c1aed8a8036af272ec58f/nn-samples/src/test/java/com/github/neuralnetworks/samples/test).

##Library structure
There are two projects:

* nn-core - contains the full implementation.
* nn-samples - contains implementations of popular datasets and 

The software design is tiered, each tier depending on the previous ones.

###Network architecture
This is the first "tier". Each network is defined by a list of layers. Each layer has a set of connections that link it to the other layers of the network, making the network a directed acyclic graph. This structure can accommodate simple feedforwad nets, but also more complex architectures like http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf. You can build your own specific network.

###Data propagation

This tier is propagating data through the network. It takes advantage of it's graph structure. There are two main base components:

* _LayerCalculator_ - propagates data through the graph. It receives target layer and input data clamped to a given layer (considered an input layer). It ensures that the data is propagated through the layers in the correct order and that all the connections in the graph are calculated. For example, during the feedforward phase of backpropagation the training data is clamped to the input layer and is propagated to the target layer (the output layer of the network). In the bp phase the output error derivative is clamped as "input" to the layer and the weights are updated using breadth-first graph traversal starting from the output layer. Essentially the role of the LayerCalculator is to provide the order, in which the network layers are calculated.
* _ConnectionCalculator_ - base class for all neuron types (sigmoid, rectifiers, convolutional etc.). After the order of calculation of the layers is determined by the _LayerCalculator_, then the list of input connections for each layer is calculated by the _ConnectionCalculator_.

####GPU
Most of the ConnectionCalculator implementations are optimized for GPU execution. Aparapi imposes some important restrictions on the code that can be executed on the GPU. The most significant are:

* only one-dimensional arrays (and variables) of primitive data types are allowed. It is not possible to use complex objects.
* only member-methods of the Aparapi Kernel class itself are allowed to be called from the GPU executable code. 

Therefore before each GPU calculation all the data is converted to one-dim arrays and primitive type variables. Because of this all Aparapi neuron types are using either _AparapiWeightedSum_ (for fully connected layers and weighted sum input functions), _AparapiSubsampling2D_ (for subsampling layers) or _AparapiConv2D_ (for convolutional layers). 
Most of the data is represented as one-dimensional array by default (for example Matrix).

###Training
All the trainers are using the _Trainer_ base class. They are optimized to run on the GPU, but you can plug-in other implementations and new training algorithms. The training procedure has training and testing phases. Each Trainer receives parameters (for example learning rate, momentum, etc) via _Properties_ (a _HashMap_). For the supported properties for each trainer please check the _TrainerFactory_ class.

###Input data
Input is provided to the neural network by the trainers via _TrainingInputProvider_ interface. Eeach _TrainingInputProvider_ provides training samples in the form of _TrainingInputData_ (default implementation is _TrainingInputDataImpl_). The input can be modified by a list of modifiers - for example _MeanInputFunction_ (for subtracting the mean value) and _ScalingInputFunction_ (scaling within a range). Currently _MnistInputProvider_ and _IrisInputProvider_ are implemented.

####Author
Ivan Vasilev (ivanvasilev [at] gmail (dot) com)

####License
[MIT License](http://opensource.org/licenses/MIT)