package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;

/**
 * Deep Belief Network
 */
public class DBN extends DNN {

    public DBN() {
	super();
    }

    /**
     * For each added layer a new RBM is created with visible layer - the hidden layer of the previous network and hidden layer - the new layer
     * @param layer
     * @param addBias
     * @return this
     */
    public DBN addLayer(Layer layer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (addLayer(layer) && getLayers().size() > 1) {
	    addNeuralNetwork(new RBM(currentOutputLayer, layer, false, addBias));
	}

	return this;
    }

    public DBN addSupervisedLayer(Layer layer, Layer dataOutputLayer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (addLayer(layer) && getLayers().size() > 1) {
	    addNeuralNetwork(new SupervisedRBM(currentOutputLayer, layer, dataOutputLayer, false, addBias));
	}

	return this;
    }
}
