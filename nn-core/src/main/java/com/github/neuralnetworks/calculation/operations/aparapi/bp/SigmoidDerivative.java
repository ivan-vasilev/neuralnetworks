package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;

public class SigmoidDerivative extends AparapiTensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	public SigmoidDerivative()
	{
		super();
	}

	public SigmoidDerivative(Tensor activations)
	{
		super(activations);
	}

	@Override
	public void run()
	{
		int id = getGlobalId();
		int activationsId = activationsStartIndex + id;
		int resultId = resultStartIndex + id;
		float activation = activations[activationsId];
		result[resultId] = result[resultId] * activation * (1 - activation);
	}
}
