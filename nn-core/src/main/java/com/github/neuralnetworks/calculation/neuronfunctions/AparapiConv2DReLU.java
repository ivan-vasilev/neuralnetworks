package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Rectified linear unit convolutional calculator
 */
public class AparapiConv2DReLU extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    public AparapiConv2DReLU() {
	super(new AparapiConv2DReLUFunction());
    }

    public static class AparapiConv2DReLUFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	@Override
	protected float activationFunction(float value) {
	    return max(0, value);
	}
    }
}
