package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * 
 * Restricted Boltzmann Machine
 * 
 */
public class RBM extends NeuralNetworkImpl {

    private GraphConnections mainConnections;
    private FullyConnected visibleBiasConnections;
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
	    Layer visibleBiasLayer = new Layer(1, new ConstantConnectionCalculator(1));
	    addLayer(visibleBiasLayer);
	    visibleBiasConnections = new FullyConnected(visibleLayer, visibleBiasLayer);
	}

	if (addHiddenBias) {
	    Layer hiddenBiasLayer = new Layer(1, new ConstantConnectionCalculator(1));
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

    @Override
    public Layer getOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }

    @Override
    public Layer getDataOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }
}
