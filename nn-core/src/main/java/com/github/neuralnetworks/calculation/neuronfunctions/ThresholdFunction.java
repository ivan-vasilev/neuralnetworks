package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Threshold binary activation function
 */
public class ThresholdFunction implements MatrixFunction {

    private float threshold;

    @Override
    public void value(Matrix inputOutput) {
	float[] elements = inputOutput.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] = elements[i] >= threshold ? 1 : 0;
	}
    }
}
