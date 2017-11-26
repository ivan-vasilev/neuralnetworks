package com.github.neuralnetworks.calculation.operations.cpu;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Threshold binary activation function
 */
public class ThresholdFunction implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	private float threshold;

	@Override
	public void value(Tensor inputOutput)
	{
		float[] elements = inputOutput.getElements();
		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = elements[i] >= threshold ? 1 : 0;
		}
	}
}
