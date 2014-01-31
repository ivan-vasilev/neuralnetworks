package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Subsampling2DConnection;

/**
 * Backpropagation for average pooling layers
 */
public class BackpropagationAveragePooling2D extends AparapiBackpropagationSubsampling2D {

    private static final long serialVersionUID = -8888670594631428090L;

    public BackpropagationAveragePooling2D(Subsampling2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int miniBatch = miniBatchSize;
	float div = 0;

	for (int i = 0; i < miniBatch; i++) {
	    div = output[getGlobalId() * miniBatch + i] / rl;
	    for (int j = 0; j < rl; j++) {
		input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i] = div;
	    }
	}
    }
}
