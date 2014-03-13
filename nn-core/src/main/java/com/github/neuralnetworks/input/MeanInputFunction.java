package com.github.neuralnetworks.input;

import com.github.neuralnetworks.calculation.neuronfunctions.MatrixFunction;
import com.github.neuralnetworks.util.Matrix;

/**
 * Mean value input function
 */
public class MeanInputFunction implements MatrixFunction {

    private static final long serialVersionUID = 1L;

    public MeanInputFunction() {
	super();
    }

    public static float getMean(Matrix input) {
	float mean = 0;
	for (float f : input.getElements()) {
	    mean += f;
	}

	return mean / input.getElements().length;
    }

    @Override
    public void value(Matrix inputOutput) {
	float mean = getMean(inputOutput);
	float[] elements = inputOutput.getElements();
	for (int i = 0; i < elements.length; i++) {
	    if (elements[i] != 0) {
		elements[i] -= mean;
	    }
	}
    }
}
