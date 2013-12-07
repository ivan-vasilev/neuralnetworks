package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void currentValuesUpdated() {
	float max = currentValues[0];
	for (int i = 1; i < regionLength; i++) {
	    if (currentValues[i] > max) {
		max = currentValues[i];
	    }
	}

	output[getGlobalId()] = max;
    }
}
