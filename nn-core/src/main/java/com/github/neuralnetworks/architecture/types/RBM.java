package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

/**
 * 
 * Restricted Boltzmann Machine
 * 
 */
public class RBM extends NeuralNetworkImpl {

    /**
     * Weights between the visible and hidden layer
     */
    private GraphConnections mainConnections;

    /**
     * Weights between visible bias layer and visible layer
     */
    private FullyConnected visibleBiasConnections;

    /**
     * Weights between hidden bias layer and hidden layer
     */
    private FullyConnected hiddenBiasConnections;

    public RBM(Layer visibleLayer, Layer hiddenLayer, boolean addVisibleBias, boolean addHiddenBias) {
	super();
	init(visibleLayer, hiddenLayer, addVisibleBias, addHiddenBias);
    }

    protected void init(Layer visibleLayer, Layer hiddenLayer, boolean addVisibleBias, boolean addHiddenBias) {
	addLayer(visibleLayer);
	addLayer(hiddenLayer);

	mainConnections = new FullyConnected(visibleLayer, hiddenLayer);

	if (addVisibleBias) {
	    Layer visibleBiasLayer = new BiasLayer();
	    addLayer(visibleBiasLayer);
	    visibleBiasConnections = new FullyConnected(visibleLayer, visibleBiasLayer);
	}

	if (addHiddenBias) {
	    Layer hiddenBiasLayer = new BiasLayer();
	    addLayer(hiddenBiasLayer);
	    hiddenBiasConnections = new FullyConnected(hiddenBiasLayer, hiddenLayer);
	}
    }

    public GraphConnections getMainConnections() {
	return mainConnections;
    }

    public GraphConnections getVisibleBiasConnections() {
	return visibleBiasConnections;
    }

    public GraphConnections getHiddenBiasConnections() {
	return hiddenBiasConnections;
    }

    public Layer getVisibleLayer() {
	return mainConnections.getInputLayer();
    }

    public Layer getHiddenLayer() {
	return mainConnections.getOutputLayer();
    }
}
