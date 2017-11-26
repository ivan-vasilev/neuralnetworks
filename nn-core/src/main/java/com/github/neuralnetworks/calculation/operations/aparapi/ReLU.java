package com.github.neuralnetworks.calculation.operations.aparapi;


/**
 * ReLU activation function
 */
public class ReLU extends AparapiTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public void run()
	{
		int id = getGlobalId() + startIndex;
		elements[id] = max(0, elements[id]);
	}
}
