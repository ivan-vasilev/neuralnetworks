package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Tanh convolutional calculator
 */
public class AparapiConv2DTanh extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    public AparapiConv2DTanh() {
	super(new AparapiConv2DTanhFunction());
    }

    public static class AparapiConv2DTanhFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	@Override
	protected float activationFunction(float value) {
	    return tan(value);
	}
    }
}
