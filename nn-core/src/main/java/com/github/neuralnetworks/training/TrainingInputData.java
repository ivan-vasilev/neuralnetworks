package com.github.neuralnetworks.training;

import java.io.Serializable;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * Training input data with target value
 * Batch Input and target are provided as matrices (each column/row is one training example)
 */
public interface TrainingInputData extends Serializable
{
	public Tensor getInput();

	public Tensor getTarget();
}
