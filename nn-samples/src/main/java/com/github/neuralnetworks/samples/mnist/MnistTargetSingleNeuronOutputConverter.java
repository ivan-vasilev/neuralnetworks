package com.github.neuralnetworks.samples.mnist;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.util.Matrix;

public class MnistTargetSingleNeuronOutputConverter implements InputConverter {

    private static final long serialVersionUID = 1L;

    @Override
    public Matrix convert(Object[] input) {
	Matrix m = new Matrix(1, input.length);

	for (int i = 0; i < input.length; i++) {
	    m.set(0, i, (int) input[i]);
	}

	return m;
    }
}
