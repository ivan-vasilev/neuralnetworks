package com.github.neuralnetworks.input.mnist;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.input.InputConverter;

public class MnistInputConverter extends InputConverter {

    @Override
    public Matrix convert(Object[] input) {
	Integer[][] arr = (Integer[][]) input;
	Matrix m = new Matrix(arr[0].length, arr.length);

	for (int i = 0; i < arr.length; i++) {
	    for (int j = 0; j < arr[i].length; j++) {
		m.set(j, i, arr[i][j]);
	    }
	}

	return m;
    }
}
