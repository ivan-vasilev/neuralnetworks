package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.io.Serializable;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Base class for loss functions
 */
public abstract class AparapiLossFunction extends Kernel implements Serializable
{
	private static final long serialVersionUID = 1L;

	protected float[] activation;
	@Constant
	protected final int activationStartPosition;
	@Constant
	protected final int activationRowStep;
	@Constant
	protected final int activationColumnStep;

	protected float[] target;
	@Constant
	protected final int targetStartPosition;
	@Constant
	protected final int targetRowStep;
	@Constant
	protected final int targetColumnStep;

	protected float[] result;
	@Constant
	protected final int resultStartPosition;
	@Constant
	protected final int resultRowStep;
	@Constant
	protected final int resultColumnStep;

	@Constant
	protected final int miniBatchSize;

	public AparapiLossFunction(Tensor activation, Tensor target, Tensor result)
	{
		if (activation.getDimensions()[0] != target.getDimensions()[0] || activation.getDimensions()[0] != result.getDimensions()[0])
		{
			throw new IllegalArgumentException("Dimensions don't match");
		}

		this.miniBatchSize = activation.getDimensions()[0];
		
		this.activation = activation.getElements();
		this.activationStartPosition = activation.getStartIndex();
		this.activationRowStep = activation.getDimensionElementsDistance(0);
		this.activationColumnStep = activation.getDimensionElementsDistance(activation.getDimensions().length - 1);

		this.target = target.getElements();
		this.targetStartPosition = target.getStartIndex();
		this.targetRowStep = target.getDimensionElementsDistance(0);
		this.targetColumnStep = target.getDimensionElementsDistance(activation.getDimensions().length - 1);

		this.result = result.getElements();
		this.resultStartPosition = result.getStartIndex();
		this.resultRowStep = result.getDimensionElementsDistance(0);
		if (result.getDimensions().length == activation.getDimensions().length)
		{
			this.resultColumnStep = result.getDimensionElementsDistance(activation.getDimensions().length - 1);
		} else
		{
			this.resultColumnStep = 0;
		}
	}

	public float[] getActivation()
	{
		return activation;
	}

	public int getActivationStartPosition()
	{
		return activationStartPosition;
	}

	public int getActivationRowStep()
	{
		return activationRowStep;
	}

	public int getActivationColumnStep()
	{
		return activationColumnStep;
	}

	public float[] getTarget()
	{
		return target;
	}

	public int getTargetStartPosition()
	{
		return targetStartPosition;
	}

	public int getTargetRowStep()
	{
		return targetRowStep;
	}

	public int getTargetColumnStep()
	{
		return targetColumnStep;
	}

	public float[] getResult()
	{
		return result;
	}

	public int getResultStartPosition()
	{
		return resultStartPosition;
	}

	public int getResultRowStep()
	{
		return resultRowStep;
	}

	public int getResultColumnStep()
	{
		return resultColumnStep;
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}
}
