package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;

public class ReLUDerivative extends AparapiTensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	public ReLUDerivative()
	{
		super();
	}

	public ReLUDerivative(Tensor activations)
	{
		super(activations);
	}

	@Override
	public void run()
	{
		if (activations[activationsStartIndex + getGlobalId()] <= 0)
		{
			result[resultStartIndex + getGlobalId()] = 0;
		}
	}
}
