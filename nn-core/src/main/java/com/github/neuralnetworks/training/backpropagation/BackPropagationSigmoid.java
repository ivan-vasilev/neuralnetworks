package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculator {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties, new AparapiBackpropSigmoid());
    }

    public static class AparapiBackpropSigmoid extends AparapiBackpropagationBase {

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeBefore(int row, int column) {
	    int id = outputBaseIndex(row, column);
	    output[id] *= outputActivation[id] * (1 - outputActivation[id]);
	}
    }
}
