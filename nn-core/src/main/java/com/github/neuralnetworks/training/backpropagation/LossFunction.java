package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * Implementations provide loss function derivative
 */
public interface LossFunction extends Serializable
{
	public float getLossFunction(Tensor activation, Tensor target);

	public void getLossFunctionDerivative(Tensor activation, Tensor target, Tensor result);
}
