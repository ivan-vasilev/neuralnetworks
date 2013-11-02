package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * 
 * Restricted Boltzmann Machine
 * 
 */
public class RBM extends NeuralNetworkImpl {

    private Connections mainConnections;
    private OneToOne visibleBiasConnections;
    private OneToOne hiddenBiasConnections;

    public RBM(Layer visibleLayer, Layer hiddenLayer, boolean addVisibleBias, boolean addHiddenBias) {
	super();

	addLayer(visibleLayer);
	addLayer(hiddenLayer);
	mainConnections = new FullyConnected(visibleLayer, hiddenLayer);

	if (addVisibleBias) {
	    Layer visibleBiasLayer = new Layer(visibleLayer.getNeuronCount(), new ConstantConnectionCalculator(1));
	    addLayer(visibleBiasLayer);
	    visibleBiasConnections = new OneToOne(visibleLayer, visibleBiasLayer);
	}

	if (addHiddenBias) {
	    Layer hiddenBiasLayer = new Layer(hiddenLayer.getNeuronCount(), new ConstantConnectionCalculator(1));
	    addLayer(hiddenBiasLayer);
	    hiddenBiasConnections = new OneToOne(hiddenBiasLayer, hiddenLayer);
	}
    }

    public Connections getMainConnections() {
	return mainConnections;
    }

    public Connections getVisibleBiasConnections() {
	return visibleBiasConnections;
    }

    public Connections getHiddenBiasConnections() {
	return hiddenBiasConnections;
    }

    public Layer getVisibleLayer() {
	return mainConnections.getInputLayer();
    }

    public Layer getHiddenLayer() {
	return mainConnections.getOutputLayer();
    }
}
