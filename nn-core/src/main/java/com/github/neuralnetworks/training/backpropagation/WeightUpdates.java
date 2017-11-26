package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

/**
 * implementations should be able to update weights of given connection
 */
public interface WeightUpdates extends Serializable
{
	public void updateWeights(float learningRate, float momentum, float l1weightDecay, float l2weightDecay);
}
