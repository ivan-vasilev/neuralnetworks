package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Subsampling2DConnection;

/**
 * Average pooling
 */
public class AparapiAveragePooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    public AparapiAveragePooling2D(Subsampling2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

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
