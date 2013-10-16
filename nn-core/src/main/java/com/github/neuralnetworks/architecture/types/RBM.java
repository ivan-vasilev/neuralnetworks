package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.neuronfunctions.ConstantInput;
import com.github.neuralnetworks.neuronfunctions.RepeaterFunction;

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

	addConnection(mainConnections = new FullyConnected(hiddenLayer, visibleLayer));

	if (addVisibleBias) {
	    addConnection(visibleBiasConnections = new OneToOne(visibleLayer, new Layer(visibleLayer.getNeuronCount(), new ConstantInput(1), new ConstantInput(1), new RepeaterFunction())));
	}

	if (addHiddenBias) {
	    addConnection(hiddenBiasConnections = new OneToOne(new Layer(hiddenLayer.getNeuronCount(), new ConstantInput(1), new ConstantInput(1), new RepeaterFunction()), hiddenLayer));
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
