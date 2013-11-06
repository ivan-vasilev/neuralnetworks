package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;

/**
 * Deep Belief Network
 */
public class DBN extends DNN {

    public DBN() {
	super();
    }

    public DBN addLayer(Layer layer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (addLayer(layer) && getLayers().size() > 0) {
	    neuralNetworks.add(new RBM(currentOutputLayer, layer, false, addBias));
	}

	return this;
    }
}
