package com.github.neuralnetworks.architecture.activation;

/**
 * this is a transfer function for a layer of neurons
 *
 */
public interface ActivationFunction {
	public float[] value(float[] input);
}
