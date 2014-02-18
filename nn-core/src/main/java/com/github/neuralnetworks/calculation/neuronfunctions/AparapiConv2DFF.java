package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Conv2DConnection;

/**
 * Base class for all feedforward convolutional functions
 */
public class AparapiConv2DFF extends AparapiConv2D {

    private static final long serialVersionUID = 5048904661076337615L;

    public AparapiConv2DFF(Conv2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

    @Override
    protected void conv(int weightsStartId, int inputStartId) {
	int id = getGlobalId();

	// calculate sum based on feature map offsets and feature map weights
	float sum = 0;

	for (int p = 0; p < miniBatchSize; p++) {
	    sum = output[id * miniBatchSize + p];

	    for (int i = 0; i < featureMapWeights; i++) {
		sum += input[(inputStartId + featureMapOffsets[i]) * miniBatchSize + p] * weights[weightsStartId + i];
	    }

	    output[id * miniBatchSize + p] = activationFunction(sum);
	}
    }

    /**
     * activation function after the convolution
     */
    protected float activationFunction(float value) {
	return value;
    }
}
