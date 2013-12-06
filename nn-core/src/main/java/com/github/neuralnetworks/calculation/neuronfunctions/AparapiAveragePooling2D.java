package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Average pooling
 */
public class AparapiAveragePooling2D extends AparapiPooling2D {

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

	output[id] = sum / regionLength;
    }
}
