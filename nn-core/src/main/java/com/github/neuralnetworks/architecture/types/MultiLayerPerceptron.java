package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

/**
 * Multi Layer Perceptron network
 */
public class MultiLayerPerceptron extends NeuralNetworkImpl {

    /**
     * A new layer is added to the output layer of the network
     * @param layer
     * @param addBias
     */
    public MultiLayerPerceptron addLayer(Layer layer, boolean addBias) {
	if (addLayer(layer) && getOutputLayer() != layer) {
	    new FullyConnected(getOutputLayer(), layer);
	}

	if (addBias && getInputLayer() != layer) {
	    ConnectionFactory.biasConnection(this, layer);
	}

	return this;
    }
}
