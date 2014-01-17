package com.github.neuralnetworks.training.backpropagation;

/**
 * Backpropagation for max pooling layers
 */
public class BackpropagationMaxPooling2D extends AparapiBackpropagationSubsampling2D {

    private static final long serialVersionUID = -8888670594631428090L;

    @Override
    protected void pool(int inputStartIndex) {
	int rl = regionLength;
	int mbs = miniBatchSize;
	int maxId = 0;
	int ffActivationId = 0;
	float max = 0;

	for (int i = 0; i < mbs; i++) {
	    ffActivationId = (inputStartIndex + featureMapOffsets[0]) * mbs + i;
	    max = ffActivation[ffActivationId];
	    for (int j = 1; j < rl; j++) {
		ffActivationId = (inputStartIndex + featureMapOffsets[j]) * mbs + i;
		float v = ffActivation[ffActivationId];
		if (v > max) {
		    maxId = ffActivationId;
		    max = v;
		}
	    }

	    input[maxId] = output[getGlobalId() * mbs + i];
	}
    }
}
