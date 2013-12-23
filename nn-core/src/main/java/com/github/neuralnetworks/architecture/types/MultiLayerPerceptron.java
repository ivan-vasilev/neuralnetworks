package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

/**
 * Multi Layer Perceptron network
 */
public class MultiLayerPerceptron extends NeuralNetworkImpl {

    public MultiLayerPerceptron addLayer(Layer layer, boolean addBias) {
	if (addLayer(layer) && getOutputLayer() != layer) {
	    new FullyConnected(getOutputLayer(), layer);
	}

	if (addBias) {
	    Layer biasLayer = new BiasLayer();
	    addLayer(biasLayer);
	    new FullyConnected(biasLayer, layer);
	}

	return this;
    }

    @Override
    public Layer getOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }

    @Override
    public Layer getDataOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }
}
