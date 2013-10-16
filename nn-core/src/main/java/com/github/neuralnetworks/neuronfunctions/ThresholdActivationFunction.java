package com.github.neuralnetworks.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

public class ThresholdActivationFunction implements ActivationFunction {

    private float threshold;

    public ThresholdActivationFunction(float threshold) {
	super();
	this.threshold = threshold;
    }

    @Override
    public void value(Matrix inputOutput) {
	float[] elements = inputOutput.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] = elements[i] >= threshold ? 1 : 0;
	}
    }
}
