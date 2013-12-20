package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for softplus layers
 */
public class BackPropagationSoftReLU extends BackPropagationConnectionCalculator {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSoftReLU(Properties properties) {
	super(properties, new AparapiBackpropSoftReLU());
    }

    public static class AparapiBackpropSoftReLU extends AparapiBackpropagationBase {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeBefore(int row, int column) {
	    output[outputBaseIndex(row, column)] = 1 / (1 + exp(-output[outputBaseIndex(row, column)]));
	}
    }
}
