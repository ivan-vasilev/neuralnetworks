package com.github.neuralnetworks.architecture;

/**
 * Connection that repeats its inputs
 */
public class RepeaterConnection extends ConnectionsImpl
{
	private static final long serialVersionUID = 1L;

	private int unitCount;

	public RepeaterConnection(Layer inputLayer, Layer outputLayer, int unitCount)
	{
		super(inputLayer, outputLayer);
		this.unitCount = unitCount;
	}

	@Override
	public int getInputUnitCount()
	{
		return unitCount;
	}

	@Override
	public int getOutputUnitCount()
	{
		return unitCount;
	}
}
