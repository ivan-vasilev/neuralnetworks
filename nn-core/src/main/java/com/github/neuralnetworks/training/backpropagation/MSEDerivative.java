package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;
import java.util.Iterator;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;

/**
 * Mean squared error derivative
 */
public class MSEDerivative implements OutputErrorDerivative {

    private static final long serialVersionUID = 1L;

    @Override
    public void getOutputErrorDerivative(Tensor activation, Tensor target, Tensor result) {
	if (!Arrays.equals(activation.getDimensions(), target.getDimensions())) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	if (result == null || !Arrays.equals(result.getDimensions(), activation.getDimensions())) {
	    result = TensorFactory.tensor(activation.getDimensions());
	}

	Iterator<Integer> activationIt = activation.iterator();
	Iterator<Integer> targetIt = target.iterator();
	Iterator<Integer> resultIt = result.iterator();

	while (resultIt.hasNext()) {
	    int activationId = activationIt.next();
	    result.getElements()[resultIt.next()] = (target.getElements()[targetIt.next()] - activation.getElements()[activationId]) * activation.getElements()[activationId] * (1 - activation.getElements()[activationId]);
	}
    }
}
