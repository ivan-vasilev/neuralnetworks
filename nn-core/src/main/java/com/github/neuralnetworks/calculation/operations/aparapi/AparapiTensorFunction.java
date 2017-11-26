package com.github.neuralnetworks.calculation.operations.aparapi;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Base aparapi tensor function
 */
public abstract class AparapiTensorFunction extends Kernel implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	protected float[] elements;
	protected int startIndex;

	@Override
	public void value(Tensor inputOutput)
	{
		if (elements == null || elements != inputOutput.getElements())
		{
			elements = inputOutput.getElements();
			startIndex = inputOutput.getStartIndex();
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, inputOutput.getSize());
	}
}
