package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;

public class SoftReLUDerivative extends AparapiTensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	public SoftReLUDerivative()
	{
		super();
	}

	public SoftReLUDerivative(Tensor activations)
	{
		super(activations);
	}

	@Override
	public void run()
	{
		int activationsId = activationsStartIndex + getGlobalId();
		int resultId = resultStartIndex + getGlobalId();
		result[resultId] = result[resultId] * (1 / (1 + exp(-activations[activationsId])));
	}
}
