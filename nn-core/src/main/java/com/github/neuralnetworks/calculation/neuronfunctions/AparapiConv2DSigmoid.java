package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Sigmoid convolutional calculator
 */
public class AparapiConv2DSigmoid extends AparapiConv2DFF {

    private static final long serialVersionUID = -7985734201416578973L;

    @Override
    protected float activationFunction(float value) {
	return 1 / (1 + exp(-value));
    }
}
