package com.github.neuralnetworks.samples.mnist;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.input.ScalingInputModifier;

public class MnistInputConverter extends InputConverter {

    public MnistInputConverter() {
	super();
	addModifier(new ScalingInputModifier(255));
    }

    @Override
    public Matrix convert(Object[] input) {
	Integer[][] arr = (Integer[][]) input;
	if (convertedInput == null || convertedInput.getColumns() != arr.length) {
	    convertedInput = new Matrix(arr[0].length, arr.length);
	}

	for (int i = 0; i < arr.length; i++) {
	    for (int j = 0; j < arr[i].length; j++) {
		convertedInput.set(j, i, arr[i][j]);
	    }
	}

	return convertedInput;
    }
}
