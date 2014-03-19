package com.github.neuralnetworks.calculation.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;

/**
 * Softmax activation function
 */
public class SoftmaxFunction extends Kernel implements MatrixFunction {

    private static final long serialVersionUID = 1L;

    private float[] values;
    private int startIndex;
    private int nextRowStep;
    private int nextColumnStep;
    private int rows;

    @Override
    public void value(Matrix inputOutput) {
	this.values = inputOutput.getElements();
	this.startIndex = inputOutput.getColumnsStartIndex();
	this.nextRowStep = inputOutput.getRowElementsDistance();
	this.nextColumnStep = inputOutput.getColumnElementsDistance();
	this.rows = inputOutput.getRows();

	Environment.getInstance().getExecutionStrategy().execute(this, inputOutput.getColumns());
    }

    @Override
    public void run() {
	float sum = 0;
	int start = startIndex;
	int r = rows;
	int nr = nextRowStep;
	int nc = nextColumnStep;
	int id = getGlobalId();

	for (int i = 0; i < r; i++) {
	    sum += values[start + i * nr + id * nc];
	}

	for (int i = 0; i < r; i++) {
	    values[start + i * nr + id * nc] /= sum;
	}
    }
}
