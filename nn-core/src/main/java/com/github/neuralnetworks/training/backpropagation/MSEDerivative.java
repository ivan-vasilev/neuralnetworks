package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;
import java.util.Iterator;

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

	Iterator<Integer> activationIt = activation.iterator();
	Iterator<Integer> targetIt = target.iterator();
	Iterator<Integer> resultIt = result.iterator();

	while (resultIt.hasNext()) {
	    int activationId = activationIt.next();
	    result.getElements()[resultIt.next()] = (target.getElements()[targetIt.next()] - activation.getElements()[activationId]) * activation.getElements()[activationId] * (1 - activation.getElements()[activationId]);
	}

	return result;
    }
}
