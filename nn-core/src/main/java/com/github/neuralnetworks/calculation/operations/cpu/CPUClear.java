package com.github.neuralnetworks.calculation.operations.cpu;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Clear tensor
 */
public class CPUClear implements TensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public void value(Tensor inputOutput)
	{
		inputOutput.forEach(i -> inputOutput.getElements()[i] = 0);
	}
}
