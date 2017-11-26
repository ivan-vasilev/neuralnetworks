package com.github.neuralnetworks.input;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * @author tmey
 */
public class OneInputProvider extends TrainingInputProviderImpl
{
	private static final long serialVersionUID = 1L;

	private final float[] input;
	private final float[] output;

	public OneInputProvider(float input[], float[] output)
	{

		this.input = input;
		this.output = output;
	}

	@Override
	public int getInputSize()
	{
		return 1;
	}

	@Override
	public int getInputDimensions()
	{
		return input.length;
	}

	@Override
	public int getTargetDimensions()
	{
		return output.length;
	}

	@Override
	public void getNextInput(Tensor nextInput)
	{
		for (int i = 0; i < nextInput.getDimensions()[0]; i++)
		{
			System.arraycopy(input, 0, nextInput.getElements(), nextInput.getStartOffset() + i * input.length, input.length);
		}
	}

	@Override
	public void getNextTarget(Tensor nextOutput)
	{
		for (int i = 0; i < nextOutput.getDimensions()[0]; i++)
		{
			System.arraycopy(output, 0, nextOutput.getElements(), nextOutput.getStartOffset() + i * output.length, output.length);
		}
	}
}
