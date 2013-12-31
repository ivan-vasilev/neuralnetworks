package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Soft ReLU convolutional calculator
 */
public class AparapiConv2DSoftReLU extends AparapiConv2D {

    private static final long serialVersionUID = -7985734201416578973L;

    @Override
    protected float activationFunction(float value) {
	return log(1 + exp(value));
    }
}
