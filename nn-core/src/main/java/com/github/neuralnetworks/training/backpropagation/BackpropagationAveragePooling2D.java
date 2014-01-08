package com.github.neuralnetworks.training.backpropagation;

/**
 * Backpropagation for average pooling layers
 */
public class BackpropagationAveragePooling2D extends AparapiBackpropagationSubsampling2D {

    private static final long serialVersionUID = -8888670594631428090L;

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int ios = inputOutputSamples;
	float div = 0;

	for (int i = 0; i < ios; i++) {
	    div = output[getGlobalId() * ios + i] / rl;
	    for (int j = 0; j < rl; j++) {
		input[(inputStartIndex + featureMapOffsets[j]) * ios + i] = div;
	    }
	}
    }
}
