package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Base class for all feedforward convolutional functions
 */
public class AparapiConv2DFF extends AparapiConv2D {

    private static final long serialVersionUID = 5048904661076337615L;

    @Override
    protected void conv(int weightsStartId, int inputStartId) {
	int id = getGlobalId();

	// calculate sum based on feature map offsets and feature map weights
	int ios = inputOutputSamples;
	int fmw = featureMapWeights;
	float sum = 0;

	for (int p = 0; p < ios; p++) {
	    sum = 0;
	    for (int i = 0, j = weightsStartId; i < fmw; i++, j++) {
		sum += input[(inputStartId + featureMapOffsets[i]) * ios + p] * weights[j];
	    }

	    output[id * ios + p] = activationFunction(sum);
	}
    }

    /**
     * activation function after the convolution
     */
    protected float activationFunction(float value) {
	return value;
    }
}
