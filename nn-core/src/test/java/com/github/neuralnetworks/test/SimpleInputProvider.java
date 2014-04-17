package com.github.neuralnetworks.test;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Simple input provider for testing purposes.
 * Training and target data are two dimensional float arrays
 */
public class SimpleInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private float[][] input;
    private float[][] target;

    public SimpleInputProvider(float[][] input, float[][] target) {
	super();

	this.input  = input;
	this.target = target;
    }

    @Override
    public int getInputSize() {
	return input.length;
    }

    @Override
    public float[] getNextInput() {
	return input[currentInput % input.length];
    }

    @Override
    public float[] getNextTarget() {
	return target[currentInput % target.length];
    }
}
