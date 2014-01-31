package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Subsampling2DConnection;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    public AparapiMaxPooling2D(Subsampling2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int miniBatch = miniBatchSize;
	float max = 0;

	for (int i = 0; i < miniBatch; i++) {
	    max = input[(inputStartIndex + featureMapOffsets[0]) * miniBatch + i];
	    for (int j = 1; j < rl; j++) {
		max = max(input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i], max);
	    }

	    output[getGlobalId() * miniBatch + i] = max;
	}
    }
}
