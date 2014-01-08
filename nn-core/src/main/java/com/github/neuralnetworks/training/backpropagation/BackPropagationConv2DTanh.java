package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Tanh Convolutional Backpropagation connection calculator for relu units
 */
public class BackPropagationConv2DTanh extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DTanh(Properties properties) {
	super(properties, new AparapiBackpropConv2DTanh());
    }

    public static class AparapiBackpropConv2DTanh extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected float activationFunctionDerivative(float value) {
	    return 1 - value * value;
	}
    }
}
