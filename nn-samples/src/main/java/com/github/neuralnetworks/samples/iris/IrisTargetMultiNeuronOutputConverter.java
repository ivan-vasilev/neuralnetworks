package com.github.neuralnetworks.samples.iris;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.TensorFactory;

public class IrisTargetMultiNeuronOutputConverter implements InputConverter {

    private static final long serialVersionUID = 1L;

    private Matrix convertedTarget;

    @Override
    public Matrix convert(Object[] input) {
	if (convertedTarget == null || convertedTarget.getColumns() != input.length) {
	    convertedTarget = TensorFactory.tensor(3, input.length);
	} else {
	    convertedTarget.forEach(i -> convertedTarget.getElements()[i] = 0);
	}

	for (int i = 0; i < input.length; i++) {
	    int val = (int) input[i];
	    convertedTarget.set(1, val, i);
	}

	return convertedTarget;
    }
}
