package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationConv2DSigmoid extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DSigmoid(Properties properties) {
	super(properties, new AparapiBackpropConv2DSigmoid());
    }

    public static class AparapiBackpropConv2DSigmoid extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected float activationFunctionDerivative(float value) {
	    return value * (1 - value);
	}
    }
}
