package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;

/**
 * RBM that has a separate data-output layer. This is an attempt to recreate http://www.cs.toronto.edu/~hinton/nipstutorial/nipstut3.pdf (slide 33)
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
