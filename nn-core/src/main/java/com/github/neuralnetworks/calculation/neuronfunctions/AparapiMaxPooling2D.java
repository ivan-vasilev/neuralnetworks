package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int ios = inputOutputSamples;
	float max = 0;

	for (int i = 0; i < ios; i++) {
	    max = input[(inputStartIndex + featureMapOffsets[0]) * ios + i];
	    for (int j = 1; j < rl; j++) {
		float v = input[(inputStartIndex + featureMapOffsets[j]) * ios + i];
		if (v > max) {
		    max = v;
		}
	    }

	    output[getGlobalId() * ios + i] = max;
	}
    }
}
