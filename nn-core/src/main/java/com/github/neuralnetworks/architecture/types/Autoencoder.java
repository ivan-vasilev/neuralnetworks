package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.Util;

/**
 * Autoencoder
 */
public class Autoencoder extends NeuralNetworkImpl {

    private Layer hiddenLayer;

    public Autoencoder(int inputUnitCount, int hiddenUnitCount, boolean addBias) {
	this(new Layer(), new Layer(), new Layer(), inputUnitCount, hiddenUnitCount, addBias);
    }

    public Autoencoder(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, int inputUnitCount, int hiddenUnitCount, boolean addBias) {
	this.hiddenLayer = hiddenLayer;

	// layers are added
	addLayer(inputLayer);
	NNFactory.addFullyConnectedLayer(this, hiddenLayer, inputUnitCount, hiddenUnitCount, addBias);
	NNFactory.addFullyConnectedLayer(this, outputLayer, hiddenUnitCount, inputUnitCount, addBias);
    }

    public Layer getHiddenBiasLayer() {
	return getHiddenLayer().getConnections().stream().map(c -> Util.getOppositeLayer(c, hiddenLayer)).filter(l -> Util.isBias(l)).findFirst().get();
    }

    public Layer getOutputBiasLayer() {
	return getOutputLayer().getConnections().stream().map(c -> Util.getOppositeLayer(c, hiddenLayer)).filter(l -> Util.isBias(l)).findFirst().get();
    }

    public Layer getHiddenLayer() {
	return hiddenLayer;
    }
}
