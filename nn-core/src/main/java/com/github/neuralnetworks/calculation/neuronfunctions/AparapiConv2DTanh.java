package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Tanh convolutional calculator
 */
public class AparapiConv2DTanh extends AparapiConv2D {

    private static final long serialVersionUID = -7985734201416578973L;

    @Override
    protected void after() {
	output[getGlobalId()] = tan(output[getGlobalId()]);
    }
}
