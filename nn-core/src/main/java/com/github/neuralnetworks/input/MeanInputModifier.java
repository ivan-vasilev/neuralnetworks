package com.github.neuralnetworks.input;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * input modifier for a mean value
 * 
 */
public class MeanInputModifier implements InputModifier {

    public MeanInputModifier() {
	super();
    }

    @Override
    public Matrix modify(Matrix input) {
	float mean = getMean(input);
	float[] elements = input.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] -= mean;
	}

	return input;
    }

    public static float getMean(Matrix input) {
	float mean = 0;
	for (float f : input.getElements()) {
	    mean += f;
	}

	return mean / input.getElements().length;
    }
}
