package com.github.neuralnetworks.calculation.operations;

import java.io.Serializable;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * Implementations provide transformations to the elements of the matrix
 */
public interface TensorFunction extends Serializable
{
	public void value(Tensor inputOutput);

	public interface TensorFunctionDerivative extends TensorFunction
	{
		public void setActivations(Tensor activations);
	}
}