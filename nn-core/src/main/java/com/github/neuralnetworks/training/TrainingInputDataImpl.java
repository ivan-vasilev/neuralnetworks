package com.github.neuralnetworks.training;

import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Training input data with target value default implementation
 */
public class TrainingInputDataImpl implements TrainingInputData
{

	private static final long serialVersionUID = 1L;

	private Tensor input;
	private Tensor target;

	public TrainingInputDataImpl()
	{
		super();
	}

	public TrainingInputDataImpl(Tensor input)
	{
		super();
		this.input = input;
	}

	public TrainingInputDataImpl(Tensor input, Tensor target)
	{
		this.input = input;
		this.target = target;
	}

	@Override
	public Tensor getInput()
	{
		return input;
	}

	public void setInput(Tensor input)
	{
		this.input = input;
	}

	@Override
	public Tensor getTarget()
	{
		return target;
	}

	public void setTarget(Matrix target)
	{
		this.target = target;
	}
}
