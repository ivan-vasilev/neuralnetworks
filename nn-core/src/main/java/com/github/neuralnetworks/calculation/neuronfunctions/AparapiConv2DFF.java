package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Base class for all feedforward convolutional functions
 */
public class AparapiConv2DFF extends AparapiConv2D {

    private static final long serialVersionUID = 5048904661076337615L;

    public AparapiConv2DFF(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	super(c, valuesProvider, targetLayer);
    }

    @Override
    protected void conv(int weightsStartId, int inputStartId, int outputStartId) {
	// calculate sum based on feature map offsets and feature map weights
	float sum = 0;

	for (int i = 0; i < miniBatchSize; i++) {
	    sum = output[outputStartId + i * outputMiniBatchDistance];

	    for (int j = 0; j < featureMapWeights; j++) {
		sum += input[inputStartId + featureMapOffsets[i * featureMapWeights + j]] * weights[weightsStartId + j];
	    }

	    output[outputStartId + i * outputMiniBatchDistance] = activationFunction(sum);
	}
    }

    /**
     * activation function after the convolution
     */
    protected float activationFunction(float value) {
	return value;
    }
}
