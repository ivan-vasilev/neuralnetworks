package com.github.neuralnetworks.calculation.operations.opencl;

import com.github.neuralnetworks.calculation.operations.TensorFunction.TensorFunctionDerivative;
import com.github.neuralnetworks.tensor.Tensor;


public abstract class OpenCLTensorFunctionDerivative extends OpenCLTensorFunction implements TensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	protected Tensor activations;

	public OpenCLTensorFunctionDerivative()
	{
		super();
	}

	public OpenCLTensorFunctionDerivative(Tensor activations)
	{
		super();
		this.activations = activations;
	}

	public Tensor getActivations()
	{
		return activations;
	}

	@Override
	public void setActivations(Tensor activations)
	{
		this.activations = activations;
	}
}
