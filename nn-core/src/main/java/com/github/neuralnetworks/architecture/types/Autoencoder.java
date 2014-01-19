package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
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
	addLayer(hiddenLayer);
	addLayer(outputLayer);

	// connections are created
	new FullyConnected(inputLayer, outputLayer);
	new FullyConnected(inputLayer, outputLayer);

	// biases are added
	if (addBias) {
	    Layer hiddenBiasLayer = new BiasLayer();
	    addLayer(hiddenBiasLayer);
	    new FullyConnected(hiddenBiasLayer, hiddenLayer);

	    Layer outputBiasLayer = new BiasLayer();
	    addLayer(outputBiasLayer);
	    new FullyConnected(outputBiasLayer, outputLayer);
	}
    }

    public BiasLayer getHiddenBiasLayer() {
	Layer hiddenLayer = getHiddenLayer();
	for (Connections c : hiddenLayer.getConnections()) {
	    Layer l = Util.getOppositeLayer(c, hiddenLayer);
	    if (l instanceof BiasLayer) {
		return (BiasLayer) l;
	    }
	}

	return null;
    }

    public BiasLayer getOutputBiasLayer() {
	Layer outputLayer = getOutputLayer();
	for (Connections c : outputLayer.getConnections()) {
	    Layer l = Util.getOppositeLayer(c, outputLayer);
	    if (l instanceof BiasLayer) {
		return (BiasLayer) l;
	    }
	}

	return null;
    }

    public Layer getHiddenLayer() {
	return hiddenLayer;
    }
}
