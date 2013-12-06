package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Stochastic pooling
 */
public class AparapiStochasticPooling2D extends AparapiPooling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    public void run() {
	int id = getGlobalId();

	// get input index
	int currentImageMapIndex = id % outputLength;
	int inputIndex = (currentImageMapIndex / outputColumns) * inputColumns + currentImageMapIndex % outputColumns;

	float sum = 0;
	for (int i = 0; i < regionLength; i++) {
	    sum += input[inputIndex + featureMapOffsets[i]];
	}

	float result = 0;
	float a = 0;
	for (int i = 0; i < regionLength; i++) {
	    a = input[inputIndex + featureMapOffsets[i]];
	    result += a * (a / sum);
	}

	output[id] = result;
    }
}
