package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;

public class TanhDerivative extends AparapiTensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	public TanhDerivative()
	{
		super();
	}

	public TanhDerivative(Tensor activations)
	{
		super(activations);
	}

	@Override
	public void run()
	{
		int id = getGlobalId();
		float activation = activations[activationsStartIndex + id];
		result[resultStartIndex + id] = result[resultStartIndex + id] * (1 - activation * activation);
	}
}
