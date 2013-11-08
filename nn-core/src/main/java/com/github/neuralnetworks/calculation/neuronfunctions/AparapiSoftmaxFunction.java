package com.github.neuralnetworks.calculation.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Util;

/**
 * softmax activation function
 */
public class AparapiSoftmaxFunction extends Kernel implements ActivationFunction {

    private float[] elements;
    private float[] sums;
    private int[] rowFinished;
    private int columns;
    private int rows;

    @Override
    public void value(Matrix inputOutput) {
	elements = inputOutput.getElements();
	columns = inputOutput.getColumns();
	rows = inputOutput.getRows();

	if (sums == null || sums.length != columns) {
	    sums = new float[columns];
	} else {
	    Util.fillArray(sums, 0);
	}

	if (rowFinished == null || rowFinished.length != columns) {
	    rowFinished = new int[columns];
	} else {
	    Util.fillArray(rowFinished, 0);
	}

	execute(Range.create2D(rows, columns));
    }

    @Override
    public void run() {
	int row = getGlobalId(0);
	int col = getGlobalId(1);
	int id = row * columns + col;
	elements[id] = exp(elements[id]);
	sums[col] += elements[id];

	// if all the rows of the current column are calculated
	rowFinished[col]++;
	if (rowFinished[col] == rows) {
	    elements[id] /= sums[col];
	}
    }
}
