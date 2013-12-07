package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Stochastic pooling
 */
public class AparapiStochasticPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void currentValuesUpdated() {
	float sum = 0;
	for (int i = 0; i < regionLength; i++) {
	    sum += currentValues[i];
	}

	float result = 0;
	if (sum > 0) {
	    float a = 0;
	    for (int i = 0; i < regionLength; i++) {
		a = currentValues[i];
		result += a * (a / sum);
	    }
	}

	output[getGlobalId()] = result;
    }
}
