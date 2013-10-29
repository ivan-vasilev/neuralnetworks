package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * 
 * Restricted Boltzmann Machine
 * 
 */
public class RBM extends NeuralNetwork {

    private Connections mainConnections;
    private OneToOne visibleBiasConnections;
    private OneToOne hiddenBiasConnections;

    public RBM(Layer visibleLayer, Layer hiddenLayer, boolean addVisibleBias, boolean addHiddenBias) {
	super();

	addConnection(mainConnections = new FullyConnected(visibleLayer, hiddenLayer));

	if (addVisibleBias) {
	    addConnection(visibleBiasConnections = new OneToOne(visibleLayer, new Layer(visibleLayer.getNeuronCount(), new ConstantConnectionCalculator(1))));
	}

	if (addHiddenBias) {
	    addConnection(hiddenBiasConnections = new OneToOne(new Layer(hiddenLayer.getNeuronCount(), new ConstantConnectionCalculator(1)), hiddenLayer));
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
	return getInputLayer();
    }

    public Layer getHiddenLayer() {
	return getOutputLayer();
    }
}
