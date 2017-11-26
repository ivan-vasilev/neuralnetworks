package com.github.neuralnetworks.input;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Scaling input function
 */
public class ScalingInputFunction implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	private Float scale;

	public ScalingInputFunction()
	{
		super();
	}

	/**
	 * All inputs are scaled according to the scale value
	 */
	public ScalingInputFunction(float scale)
	{
		super();
		this.scale = scale;
	}

	public float getScale()
	{
		return scale;
	}

	public void setScale(float scale)
	{
		this.scale = scale;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		int mb = inputOutput.getDimensions()[0];
		int length = inputOutput.getSize() / inputOutput.getDimensions()[0];

		if (scale == null)
		{
			for (int i = 0; i < mb; i++)
			{
				float max = Float.MIN_VALUE;
				float min = Float.MAX_VALUE;
				for (int j = 0; j < length; j++)
				{
					max = Math.max(max, inputOutput.getElements()[inputOutput.getStartIndex() + i * length + j]);
					min = Math.min(min, inputOutput.getElements()[inputOutput.getStartIndex() + i * length + j]);
				}

				for (int j = 0; j < length; j++)
				{
					inputOutput.getElements()[inputOutput.getStartIndex() + i * length + j] = (inputOutput.getElements()[inputOutput.getStartIndex() + i * length + j] - min) / (max - min);
				}
			}
		} else
		{
			for (int i = 0; i < mb; i++)
			{

				for (int j = 0; j < length; j++)
				{
					inputOutput.getElements()[inputOutput.getStartIndex() + i * length + j] /= scale;
				}
			}
		}
	}
}
