package com.github.neuralnetworks.input.mnist;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.input.InputConverter;

public class MnistTargetConverter extends InputConverter {

    @Override
    public Matrix convert(Object[] input) {
	Matrix m = new Matrix(10, input.length);

	for (int i = 0; i < input.length; i++) {
	    int val = (int) input[i];
	    m.set(val, i, val);
	}

	return m;
    }
}
