package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.util.Matrix;

/**
 * Mean squared error derivative
 */
public class MSEDerivative implements OutputErrorDerivative {

    private static final long serialVersionUID = 1L;

    private Matrix result;

    @Override
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target) {
	if (activation.getElements().length != target.getElements().length || activation.getColumns() != target.getColumns()) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	if (result == null || result.getElements().length != activation.getElements().length) {
	    result = new Matrix(activation);
	}

	for (int i = 0; i < activation.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - activation.getElements()[i]) * activation.getElements()[i] * (1 - activation.getElements()[i]);
	}

	return result;
    }
}
