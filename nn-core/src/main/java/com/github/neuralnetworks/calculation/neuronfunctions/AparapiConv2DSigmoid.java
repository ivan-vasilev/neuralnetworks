package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Sigmoid convolutional calculator
 */
public class AparapiConv2DSigmoid extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    public AparapiConv2DSigmoid() {
	super(new AparapiConv2DSigmoidFunction());
    }

    public static class AparapiConv2DSigmoidFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	@Override
	protected float activationFunction(float value) {
	    return 1 / (1 + exp(-value));
	}
    }
}
