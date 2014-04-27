package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Softmax layer
 */
public class AparapiSoftmax extends AparapiWeightedSumConnectionCalculator {

    private static final long serialVersionUID = 1L;
    public AparapiSoftmax() {
	super();
	addActivationFunction(new SoftmaxFunction());
    }
}
