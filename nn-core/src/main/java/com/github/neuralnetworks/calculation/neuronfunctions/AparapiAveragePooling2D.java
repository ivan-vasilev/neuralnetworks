package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Average pooling
 */
public class AparapiAveragePooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int miniBatch = miniBatchSize;
	float sum = 0;

	for (int i = 0; i < miniBatch; i++) {
	    sum = 0;
	    for (int j = 0; j < rl; j++) {
		sum += input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i];
	    }
	    
	    output[getGlobalId() * miniBatch + i] = sum / rl;
	}
    }
}
