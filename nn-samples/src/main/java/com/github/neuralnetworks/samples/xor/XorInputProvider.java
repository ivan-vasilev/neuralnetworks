package com.github.neuralnetworks.samples.xor;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * XOR input provider
 */
public class XorInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private float[] input;
    private float[] target;

    public XorInputProvider() {
	super();
	input = new float[2];
	target = new float[2];
    }

    @Override
    public int getInputSize() {
	return 4;
    }

    @Override
    public float[] getNextInput() {
	switch (currentInput % 4) {
	case 0:
	    input[0] = input[1] = 0;
	    break;
	case 1:
	    input[0] = 0;
	    input[1] = 1;
	    break;
	case 2:
	    input[0] = 1;
	    input[1] = 0;
	    break;
	case 3:
	    input[0] = input[1] = 1;
	    break;
	}

	return input;
    }

    @Override
    public float[] getNextTarget() {
	switch (currentInput % 4) {
	case 0:
	    target[0] = 0;
	    break;
	case 1:
	    target[0] = 1;
	    break;
	case 2:
	    target[0] = 1;
	    break;
	case 3:
	    target[0] = 0;
	    break;
	}

	return target;
    }
}
