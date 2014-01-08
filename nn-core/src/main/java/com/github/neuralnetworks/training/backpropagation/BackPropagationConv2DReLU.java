package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator for relu units
 */
public class BackPropagationConv2DReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DReLU(Properties properties) {
	super(properties, new AparapiBackpropConv2DReLU());
    }

    public static class AparapiBackpropConv2DReLU extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected float activationFunctionDerivative(float value) {
	    if (value > 0) {
		return value;
	    } else {
		return 0;
	    }
	}
    }
}
