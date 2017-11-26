package com.github.neuralnetworks.calculation.operations.aparapi;


/**
 * Sigmoid activation function
 */
public class Sigmoid extends AparapiTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public void run()
	{
		int id = getGlobalId() + startIndex;
		elements[id] = 1 / (1 + exp(-elements[id]));
	}
}
