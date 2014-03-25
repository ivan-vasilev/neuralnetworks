package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;

import com.github.neuralnetworks.util.Tensor;

/**
 * Mean squared error derivative
 */
public class MSEDerivative implements OutputErrorDerivative {

    private static final long serialVersionUID = 1L;

    private Tensor result;

    @Override
    public Tensor getOutputErrorDerivative(Tensor activation, Tensor target) {
	if (!Arrays.equals(activation.getDimensions(), target.getDimensions())) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	if (result == null || result.getElements().length != activation.getElements().length) {
	    result = new Tensor(activation.getDimensions());
	}

	for (int i = 0; i < activation.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - activation.getElements()[i]) * activation.getElements()[i] * (1 - activation.getElements()[i]);
	}

	return result;
    }
}
