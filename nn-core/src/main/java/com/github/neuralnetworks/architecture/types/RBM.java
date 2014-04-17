package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.Util;

/**
 * 
 * Restricted Boltzmann Machine
 * 
 */
public class RBM extends NeuralNetworkImpl {

    private static final long serialVersionUID = 1L;

    public RBM() {
	super();
    }

    public RBM(Layer visibleLayer, Layer hiddenLayer, int visibleUnitCount, int hiddenUnitCount, boolean addVisibleBias, boolean addHiddenBias) {
	super();
	init(visibleLayer, hiddenLayer, visibleUnitCount, hiddenUnitCount, addVisibleBias, addHiddenBias);
    }

    protected void init(Layer visibleLayer, Layer hiddenLayer, int visibleUnitCount, int hiddenUnitCount, boolean addVisibleBias, boolean addHiddenBias) {
	addLayer(visibleLayer);
	addLayer(hiddenLayer);

	new FullyConnected(visibleLayer, hiddenLayer, visibleUnitCount, hiddenUnitCount);

	if (addVisibleBias) {
	    Layer visibleBiasLayer = new Layer();
	    addLayer(visibleBiasLayer);
	    new FullyConnected(visibleBiasLayer, visibleLayer, 1, visibleUnitCount);
	}

	if (addHiddenBias) {
	    Layer hiddenBiasLayer = new Layer();
	    addLayer(hiddenBiasLayer);
	    new FullyConnected(hiddenBiasLayer, hiddenLayer, 1, hiddenUnitCount);
	}
    }

    public FullyConnected getMainConnections() {
	return (FullyConnected) getConnections().stream().filter(c -> c.getInputLayer() == getInputLayer() && c.getOutputLayer() == getOutputLayer()).findFirst().orElse(null);
    }

    public FullyConnected getVisibleBiasConnections() {
	return (FullyConnected) getConnections().stream().filter(c -> c.getOutputLayer() == getInputLayer() && Util.isBias(c.getInputLayer())).findFirst().orElse(null);
    }

    public FullyConnected getHiddenBiasConnections() {
	return (FullyConnected) getConnections().stream().filter(c -> c.getOutputLayer() == getOutputLayer() && Util.isBias(c.getInputLayer())).findFirst().orElse(null);
    }

    public Layer getVisibleLayer() {
	return getInputLayer();
    }

    public Layer getHiddenLayer() {
	return getOutputLayer();
    }
}
