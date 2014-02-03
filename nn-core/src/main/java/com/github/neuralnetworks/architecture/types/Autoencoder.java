package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.Util;

/**
 * Autoencoder
 */
public class Autoencoder extends NeuralNetworkImpl {

    private Layer hiddenLayer;

    public Autoencoder(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, boolean addBias) {
	this.hiddenLayer = hiddenLayer;

	// layers are added
	addLayer(inputLayer);
	NNFactory.addFullyConnectedLayer(this, hiddenLayer, addBias);
	NNFactory.addFullyConnectedLayer(this, outputLayer, addBias);
    }

    public Layer getHiddenBiasLayer() {
	Layer hiddenLayer = getHiddenLayer();
	for (Connections c : hiddenLayer.getConnections()) {
	    Layer l = Util.getOppositeLayer(c, hiddenLayer);
	    if (Util.isBias(l)) {
		return l;
	    }
	}

	return null;
    }

    public Layer getOutputBiasLayer() {
	Layer outputLayer = getOutputLayer();
	for (Connections c : outputLayer.getConnections()) {
	    Layer l = Util.getOppositeLayer(c, outputLayer);
	    if (Util.isBias(l)) {
		return l;
	    }
	}

	return null;
    }

    public Layer getHiddenLayer() {
	return hiddenLayer;
    }
}
