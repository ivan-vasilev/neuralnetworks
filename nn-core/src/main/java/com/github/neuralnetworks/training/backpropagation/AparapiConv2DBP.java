package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;

/**
 * BackPropagation base function for convolutional layers
 */
public class AparapiConv2DBP extends AparapiConv2D {

    private static final long serialVersionUID = -345286029645674230L;

    @Override
    protected void conv(int weightsStartId, int inputStartId) {
	int id = getGlobalId();

	int ios = inputOutputSamples;
	int fmw = featureMapWeights;
	float activation = 0;

	for (int p = 0; p < ios; p++) {
	    activation = activationFunctionDerivative(output[id * ios + p]);;
	    output[id * ios + p] = activation;

	    for (int i = 0, j = weightsStartId; i < fmw; i++, j++) {
		input[(inputStartId + featureMapOffsets[i]) * ios + p] += activation * weights[j];
	    }
	}
    }

    /**
     * Derivative of the FF activation function
     * 
     * @param value
     * @return
     */
    protected float activationFunctionDerivative(float value) {
	return value;
    }
}
