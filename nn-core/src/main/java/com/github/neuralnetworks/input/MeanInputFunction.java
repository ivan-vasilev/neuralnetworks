package com.github.neuralnetworks.input;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Mean value input function
 */
public class MeanInputFunction implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	public MeanInputFunction()
	{
		super();
	}

	public static float getMean(Tensor input)
	{
		float mean = 0;
		for (float f : input.getElements())
		{
			mean += f;
		}

		return mean / input.getSize();
	}

	@Override
	public void value(Tensor inputOutput)
	{
		float mean = getMean(inputOutput);
		float[] elements = inputOutput.getElements();
		for (int i = 0; i < elements.length; i++)
		{
			if (elements[i] != 0)
			{
				elements[i] -= mean;
			}
		}
	}
}
