package com.github.neuralnetworks.training.random;

import java.io.Serializable;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * Base interface for random initialization of arrays
 */
public interface RandomInitializer extends Serializable
{
	public void initialize(Tensor t);

	/**
	 *
	 * @return true only if it was possible, else false
	 */
	public boolean reset();
}
