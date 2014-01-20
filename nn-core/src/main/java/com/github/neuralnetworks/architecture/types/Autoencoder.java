package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.util.Util;

/**
 * Autoencoder
 */
public class Autoencoder extends MultiLayerPerceptron {

    private Layer hiddenLayer;

    public Autoencoder(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, boolean addBias) {
	this.hiddenLayer = hiddenLayer;

	// layers are added
	addLayer(inputLayer);
	addLayer(hiddenLayer, addBias);
	addLayer(outputLayer, addBias);
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
