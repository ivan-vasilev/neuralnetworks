package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculator implements OutputErrorDerivative {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties, new AparapiBackpropSigmoidByRows(), new AparapiBackpropSigmoidByColumns());
    }

    @Override
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target) {
	if (activation.getElements().length != target.getElements().length || activation.getColumns() != target.getColumns()) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	Matrix result = new Matrix(activation);
	for (int i = 0; i < activation.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - activation.getElements()[i]) * activation.getElements()[i] * (1 - activation.getElements()[i]);
	}

	return result;
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
