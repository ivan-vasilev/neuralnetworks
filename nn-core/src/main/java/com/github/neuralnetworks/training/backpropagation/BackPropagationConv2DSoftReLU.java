package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator for softplus layers
 */
public class BackPropagationConv2DSoftReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DSoftReLU(Properties properties) {
	super(properties, new AparapiBackpropConv2DSoftReLU());
    }

    public static class AparapiBackpropConv2DSoftReLU extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected float activationFunctionDerivative(float value) {
	    return (1 / (1 + exp(-value)));
	}
    }
}
