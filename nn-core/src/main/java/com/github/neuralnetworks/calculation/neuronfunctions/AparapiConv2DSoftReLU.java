package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Softplus convolutional calculator
 */
public class AparapiConv2DSoftReLU extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    public AparapiConv2DSoftReLU() {
	super(new AparapiConv2DSoftReLUFunction());
    }

    public static class AparapiConv2DSoftReLUFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	@Override
	protected float activationFunction(float value) {
	    return log(1 + exp(value));
	}
    }
}
