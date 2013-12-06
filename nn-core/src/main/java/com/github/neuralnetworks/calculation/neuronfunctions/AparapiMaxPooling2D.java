package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends AparapiPooling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    public void run() {
	int id = getGlobalId();

	// get input index
	int currentImageMapIndex = id % outputLength;
	int inputIndex = (currentImageMapIndex / outputColumns) * inputColumns + currentImageMapIndex % outputColumns;

	float max = input[inputIndex];
	for (int i = 0; i < regionLength; i++) {
	    if (input[inputIndex + featureMapOffsets[i]] > max) {
		max = input[inputIndex + featureMapOffsets[i]];
	    }
	}

	output[id] = max;
    }
}
