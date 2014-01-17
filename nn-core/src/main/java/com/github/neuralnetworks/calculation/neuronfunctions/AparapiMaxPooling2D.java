package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int miniBatch = miniBatchSize;
	float max = 0;

	for (int i = 0; i < miniBatch; i++) {
	    max = input[(inputStartIndex + featureMapOffsets[0]) * miniBatch + i];
	    for (int j = 1; j < rl; j++) {
		float v = input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i];
		if (v > max) {
		    max = v;
		}
	    }

	    output[getGlobalId() * miniBatch + i] = max;
	}
    }
}
