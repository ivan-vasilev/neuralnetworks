package com.github.neuralnetworks.input;

import java.util.Arrays;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;

public class MultipleNeuronsSimpleOutputError implements OutputError
{
	private static final long serialVersionUID = 1L;

	private int correct;
	private int incorrect;

	public MultipleNeuronsSimpleOutputError()
	{
		super();
		reset();
	}

	@Override
	public void addItem(Tensor newtorkOutput, Tensor targetOutput)
	{
		Matrix target = (Matrix) targetOutput;
		Matrix actual = (Matrix) newtorkOutput;

		if (!Arrays.equals(actual.getDimensions(), target.getDimensions()))
		{
			throw new IllegalArgumentException("Dimensions don't match");
		}

		for (int i = 0; i < target.getRows(); i++)
		{
			boolean hasDifferentValues = false;
			for (int j = 0; j < actual.getColumns(); j++)
			{
				if (actual.get(i, j) != actual.get(i, 0))
				{
					hasDifferentValues = true;
					break;
				}
			}

			if (hasDifferentValues)
			{
				int targetPos = 0;
				for (int j = 0; j < target.getColumns(); j++)
				{
					if (target.get(i, j) == 1)
					{
						targetPos = j;
						break;
					}
				}

				int outputPos = 0;
				float max = actual.get(i, 0);
				for (int j = 0; j < actual.getColumns(); j++)
				{
					if (actual.get(i, j) > max)
					{
						max = actual.get(i, j);
						outputPos = j;
					}
				}

				if (targetPos == outputPos)
				{
					correct++;
				} else
				{
					incorrect++;
				}
			} else
			{
				incorrect++;
			}
		}
	}

	@Override
	public float getTotalNetworkError()
	{
		return  getTotalInputSize() > 0 ? ((float) getTotalErrorSamples()) / getTotalInputSize() : 0;
	}

	@Override
	public int getTotalErrorSamples()
	{
		return incorrect;
	}

	@Override
	public int getTotalInputSize()
	{
		return incorrect + correct;
	}

	@Override
	public void reset()
	{
		correct = incorrect = 0;
	}
}
