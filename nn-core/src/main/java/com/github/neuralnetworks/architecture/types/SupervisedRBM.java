package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;

/**
 * RBM with extra layer
 *
 */
public class SupervisedRBM extends RBM {

    private Layer dataOutputLayer;

    public SupervisedRBM(Layer visibleLayer, Layer hiddenLayer, Layer dataOutputLayer, boolean addVisibleBias, boolean addHiddenBias) {
	super(visibleLayer, hiddenLayer, addVisibleBias, addHiddenBias);
	addLayer(dataOutputLayer);
	this.dataOutputLayer = dataOutputLayer;
    }

    @Override
    public Layer getDataOutputLayer() {
	return dataOutputLayer;
    }
}
