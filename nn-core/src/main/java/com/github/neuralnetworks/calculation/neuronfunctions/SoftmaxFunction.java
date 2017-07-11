package com.github.neuralnetworks.calculation.neuronfunctions;

import com.aparapi.Kernel;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Softmax activation function
 */
public class SoftmaxFunction extends Kernel implements TensorFunction {

    private static final long serialVersionUID = 1L;

    private float[] values;
    private int startIndex;
    private int nextRowStep;
    private int nextColumnStep;
    private int rows;

    @Override
    public void value(Tensor inputOutput) {
	Matrix io = (Matrix) inputOutput;

	this.values = io.getElements();
	this.startIndex = io.getStartIndex();
	this.nextRowStep = io.getRowElementsDistance();
	this.nextColumnStep = io.getColumnElementsDistance();
	this.rows = io.getRows();

	Environment.getInstance().getExecutionStrategy().execute(this, io.getColumns());
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
            values[start + i * nr + id * nc] = exp(values[start + i * nr + id * nc]);
	    sum += values[start + i * nr + id * nc];	
	}

	for (int i = 0; i < r; i++) {
	    values[start + i * nr + id * nc] /= sum;
	}
    }
}
