package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.TensorFunction.TensorFunctionDerivative;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Base class for derivatives
 */
public abstract class AparapiTensorFunctionDerivative extends Kernel implements TensorFunctionDerivative
{
	private static final long serialVersionUID = 1L;

	protected float[] activations;
	protected int activationsStartIndex;

	protected float[] result;
	protected int resultStartIndex;

	public AparapiTensorFunctionDerivative()
	{
		super();
	}

	public AparapiTensorFunctionDerivative(Tensor activations)
	{
		setActivations(activations);
	}

	@Override
	public void setActivations(Tensor activations) {
		this.activations = activations.getElements();
		this.activationsStartIndex = activations.getStartIndex();
	}

	@Override
	public void value(Tensor inputOutput)
	{
		if (result == null || result != inputOutput.getElements())
		{
			result = inputOutput.getElements();
			resultStartIndex = inputOutput.getStartIndex();
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, inputOutput.getSize());
	}
}
