package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Stochastic pooling
 */
public class AparapiStochasticPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void pool(int inputStartIndex) {
	int miniBatch = miniBatchSize;
	int rl = regionLength;
	float sum = 0;
	float result = 0;
	float a = 0;

	for (int i = 0; i < miniBatch; i++) {
	    sum = 0;
	    result = 0;

	    for (int j = 0; j < rl; j++) {
		sum += input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i];
	    }

	    if (sum > 0) {
		a = 0;
		for (int j = 0; j < rl; j++) {
		    a = input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i];
		    result += a * (a / sum);
		}
	    }

	    output[getGlobalId() * miniBatch + i] = result;
	}
    }
}
