package com.github.neuralnetworks.architecture.types;

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


    public RBM(int visibleUnitCount, int hiddenUnitCount, boolean addVisibleBias, boolean addHiddenBias) {
	super();
	init(new Layer(), new Layer(), visibleUnitCount, hiddenUnitCount, addVisibleBias, addHiddenBias);
    }

    public RBM(Layer visibleLayer, Layer hiddenLayer, int visibleUnitCount, int hiddenUnitCount, boolean addVisibleBias, boolean addHiddenBias) {
	super();
	init(visibleLayer, hiddenLayer, visibleUnitCount, hiddenUnitCount, addVisibleBias, addHiddenBias);
    }

    protected void init(Layer visibleLayer, Layer hiddenLayer, int visibleUnitCount, int hiddenUnitCount, boolean addVisibleBias, boolean addHiddenBias) {
	addLayer(visibleLayer);
	addLayer(hiddenLayer);

	mainConnections = new FullyConnected(visibleLayer, hiddenLayer, visibleUnitCount, hiddenUnitCount);

	if (addVisibleBias) {
	    Layer visibleBiasLayer = new Layer();
	    addLayer(visibleBiasLayer);
	    visibleBiasConnections = new FullyConnected(visibleBiasLayer, visibleLayer, 1, visibleUnitCount);
	}

	if (addHiddenBias) {
	    Layer hiddenBiasLayer = new Layer();
	    addLayer(hiddenBiasLayer);
	    hiddenBiasConnections = new FullyConnected(hiddenBiasLayer, hiddenLayer, 1, hiddenUnitCount);
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
