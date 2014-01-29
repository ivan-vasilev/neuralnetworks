package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Softmax activation function
 */
public class SoftmaxFunction implements MatrixFunction {

    @Override
    public void value(Matrix inputOutput) {
	int rows = inputOutput.getRows();
	int columns = inputOutput.getColumns();
	float sum = 0;

	for (int i = 0; i < columns; i++) {
	    sum = 0;

	    for (int j = 0; j < rows; j++) {
		sum += inputOutput.get(j, i);
	    }

	    for (int j = 0; j < rows; j++) {
		inputOutput.set(j, i, sum != 0 ? inputOutput.get(j,  i) / sum : 0);
	    }
	}
    }
}
