package com.github.neuralnetworks.input;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Simple input provider for testing purposes.
 * Training and target data are two dimensional float arrays
 */
public class SimpleInputProvider extends TrainingInputProviderImpl
{

	private static final long serialVersionUID = 1L;

	private float[][] input;
	private float[][] target;

	public SimpleInputProvider(float[][] input)
	{
		super();
		this.input = input;
	}

	public SimpleInputProvider(float[][] input, float[][] target)
	{
		super();
		this.input = input;
		this.target = target;

		if (input.length != target.length)
		{
			throw new RuntimeException("Target and input minibatch size does not match");
		}
	}

	@Override
	public int getInputSize()
	{
		return input.length;
	}

	@Override
	public void getNextInput(Tensor tin)
	{
		if (tin != null)
		{
			int mb = tin.getDimensions()[0];
			for (int i = 0; i < mb; i++)
			{
				int current = (i + currentInput) % input.length;
				System.arraycopy(input[current], 0, tin.getElements(), tin.getStartIndex() + input[current].length * i, input[current].length);
			}
		}
	}

	@Override
	public void getNextTarget(Tensor tt)
	{
		if (tt != null)
		{
			int mb = tt.getDimensions()[0];
			for (int i = 0; i < mb; i++)
			{
				int current = (currentInput + i) % target.length;
				System.arraycopy(target[current], 0, tt.getElements(), tt.getStartIndex() + target[current].length * i, target[current].length);
			}
		}
	}

	public float[][] getInput()
	{
		return input;
	}

	public float[][] getTarget()
	{
		return target;
	}

	@Override
	public int getInputDimensions()
	{
		return input[0].length;
	}

	@Override
	public int getTargetDimensions()
	{
		if (target != null)
		{
			return target[0].length;
		}

		return 0;
	}
}
