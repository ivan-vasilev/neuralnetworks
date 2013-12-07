package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Average pooling
 */
public class AparapiAveragePooling2D extends AparapiSubsampling2D {

    private static final long serialVersionUID = -2393526660090301257L;

    @Override
    protected void currentValuesUpdated() {
	float sum = 0;
	for (int i = 0; i < regionLength; i++) {
	    sum += currentValues[i];
	}

	output[getGlobalId()] = sum / regionLength;
    }
}
