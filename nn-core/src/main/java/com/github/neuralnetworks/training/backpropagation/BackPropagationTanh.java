package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for tanh layers
 */
public class BackPropagationTanh extends BackPropagationConnectionCalculator {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationTanh(Properties properties) {
	super(properties, new AparapiBackpropTanh());
    }

    public static class AparapiBackpropTanh extends AparapiBackpropagationBase {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeBefore(float activation, float error, int outputId) {
	    output[outputId] = error * - error * activation * activation;
	}
    }
}
