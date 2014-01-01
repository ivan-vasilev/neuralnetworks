Deep Neural Networks with GPU support
=====================================

This is a Java implementation of some of the algorithms for training deep neural networks. GPU support is provided via the OpenCL and Aparapi.
The architecture is designed with modularity, extensibility and pluggability in mind.
Supported networks are Multi Layer Perceptron, Autoencoders, Restricted Boltzmann Machines, Convolutional (and subsampling) networks, Stacked Autoencoders, Deep Belief Nets.
To build this project you need gradle (http://www.gradle.org/) or maven (http://maven.apache.org/, to be deprecated).

There are two projects:
- nn-core - the full implementation is here.
- nn-samples - various examples and InputProviders for popular training data sets (currently only MNIST is implemented).

The design is tiered, each tier depending on the previous ones.

Architecture
------------
Each NeuralNetwork (default NeuralNetworkImpl) is defined by a list of Layers. Each layer has a set of Connections (default ConnectionsImpl) that link it to the other layers of the network. Each neural network is essentially a directed acyclic graph. This structure can accommodate simple feedforwad networks, but also more complex architectures like http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf.

###Layer types
- Default Layer - has a neuron count property and default ConnectionCalculator (explained in the next section).
- Bias Layer - bias layer.
- ConvGridLayer - Convolutional 2D layer that contains information about the number of feature maps and width and height of each feature map.

###Connections between layers
- FullyConnected - fully connected layers of neurons. The weights are matrices (of the type Matrix).
- Conv2DConnection - convolutional connection between two ConvGridLayers. Contains a list of filters for each feature map.
- Subsampling2DConnection - subsampling connection between two ConvGridLayers. Contains subsampling region dimensions.

###Neural Networks
- MultiLayerPerceptron - Multilayer Perceptron.
- RBM - Restricted Boltzmann Machine.
- Autoencoder, Denoising autoencoder
- DBN - Deep Belief Network.
- StackedAutoencoder - Stacked Autoencoder.
- Convolutional Networks - essentially MultiLayerPerceptrons with convolutional and subsampling layers and connections.

Calculation
-----------

This tier is propagating data through the network. It takes advantage of the graph structure of the network. There are two main base components:
- LayerCalculator (defualt implementation LayerCalculatorImpl) - propagates data through the graph. It receives target layer and input data clamped to a given layer (considered an input layer). It ensures that the data is propagated through the layers in the correct order and that all the connections in the graph are calculated. For example, in Backpropagation feedforward phase the training data is clamped to the input layer and the target layer is the network output layer. In the bp phase the output error derivative is clamped as "input" to the network output layer and the target layer is the network input layer.
- ConnectionCalculator (default implementation ConnectionCalculatorImpl) - base class for all neuron types (sigmoid, rectifiers, convolutional etc.). After the order of calculation of the layers is determined by LayerCalculator, then the list of input connections for each layer is calculated by the ConnectionCalculator.

###GPU

Most of the ConnectionCalculator implementations are optimized for GPU execution. Aparapi imposes some important restrictions on the code that can be executed on the GPU. The most significant are:
- only one-dimensional arrays (and variables) of primitive data types are allowed. It is not possible to use complex objects.
- only member-methods of the Aparapi Kernel class itself are allowed to be called from the GPU executable code. 

Therefore before each GPU calculation all the data is converted to one-dim arrays and primitive type variables. Because of this all Aparapi neuron types are using either AparapiWeightedSum (for fully connected layers and weighted sum input functions), AparapiSubsampling2D (for subsampling layers) or AparapiConv2D (for convolutional layers). 
Most of the data is represented as one-dimensional array by default (for example Matrix).

###Neuron types based on weighted sum

- AparapiSigmoid - sigmoid activation function.
- AparapiSoftReLU - softplus activation function.
- AparapiTanh - tanh activation function.
- AparapiSoftmax - softmax layer.
- AparapiStochasticBinary - stochastic binary activation function (for RBMs).

###Convolutional layer types

- AparapiConv2DSigmoid - sigmoid convolutional layer.
- AparapiConv2DSoftReLU - softmax convolutional layer.
- AparapiConv2DTanh - tanh convolutional layer.

###Subsampling layer types

- AparapiAveragePooling2D - average pooling
- AparapiMaxPooling2D - max pooling
- AparapiStochasticPooling2D - stochastic pooling (http://techtalks.tv/talks/stochastic-pooling-for-regularization-of-deep-convolutional-neural-networks/58106/)

Training
--------

All the trainers are using the Trainer base class. The implementations are optimized to run on the GPU, but all the trainers are designed in such a way that another implementation can be plugged.The training procedure has training and testing phases. Each Trainer receives parameters (for example learning rate, momentum, etc) via Properties (a HashMap). For the supported properties for each trainer please check the TrainerFactory class.

###Supported training algorithms
- BackPropagationTrainer - Backpropagation. Aparapi implementation is provided (check the package) for sigmoid, tanh, softplus functions and denoising autoencoders. Backpropagation is not yet working for convolutional and subsampling layers.
- CDAparapiTrainer - GPU optimized Contrastive Divergence via Aparapi.
- PCDAparapiTrainer -GPU optimized Persistent Contrastive Divergence.
- GreedyLayerDNNTrainer - For Deep networks training. Each "layer" (child network) is trained by it's own trainer. After a level is finished, the next level starts training. The training inputs for the current level are propagated through the previous levels first.

Contrastive Divergence and Deep training has been implemented following the guidelines in 
http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239 and http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf.

###Input data
Input is provided to the neural network by the trainers via TrainingInputProvider interface. Eeach TrainingInputProvider provides training examples in the form of TrainingInputData (default implementation is TrainingInputDataImpl). The input can be modified by a list of InputModifiers - for example MeanInputModifier (for subtracting the mean value) and ScalingInputModifier (scaling within a range). Currently MnistInputProvider is implemented.
