package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculator {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties, new AparapiBackpropSigmoidByRows(), new AparapiBackpropSigmoidByColumns());
    }

    public static class AparapiBackpropSigmoidByRows extends AparapiBackpropagationBaseByRows {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeBefore(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);
	}
    }

    public static class AparapiBackpropSigmoidByColumns extends AparapiBackpropagationBaseByColumns {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeBefore(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);
	}
    }
}
