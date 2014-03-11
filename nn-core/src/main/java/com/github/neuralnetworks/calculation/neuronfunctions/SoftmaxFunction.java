package com.github.neuralnetworks.calculation.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Environment;

/**
 * Softmax activation function
 */
public class SoftmaxFunction extends Kernel implements MatrixFunction {

    private static final long serialVersionUID = 1L;

    private float[] values;
    private int rows;
    private int columns;

    @Override
    public void value(Matrix inputOutput) {
	this.values = inputOutput.getElements();
	this.rows = inputOutput.getRows();
	this.columns = inputOutput.getColumns();

	Environment.getInstance().getExecutionStrategy().execute(this, columns);
    }

    @Override
    public void run() {
	float sum = 0;
	int r = rows;
	int c = columns;
	int id = getGlobalId();

	for (int i = 0; i < r; i++) {
	    sum += values[i * c + id];
	}

	for (int i = 0; i < r; i++) {
	    values[i * c + id] /= sum;
	}
    }
}
