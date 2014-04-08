package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.Util;

/**
 * Autoencoder
 */
public class Autoencoder extends NeuralNetworkImpl {

    private static final long serialVersionUID = 1L;

    public Autoencoder() {
	super();
    }

    public Layer getHiddenBiasLayer() {
	Layer hiddenLayer = getHiddenLayer();
	return hiddenLayer.getConnections().stream().map(c -> Util.getOppositeLayer(c, hiddenLayer)).filter(l -> Util.isBias(l)).findFirst().orElse(null);
    }

    public Layer getOutputBiasLayer() {
	Layer outputLayer = getOutputLayer();
	return outputLayer.getConnections().stream().map(c -> Util.getOppositeLayer(c, outputLayer)).filter(l -> Util.isBias(l)).findFirst().orElse(null);
    }

    public Layer getHiddenLayer() {
	return getLayers().stream().filter(l -> l != getOutputLayer() && l != getInputLayer()).findFirst().orElse(null);
    }
}
