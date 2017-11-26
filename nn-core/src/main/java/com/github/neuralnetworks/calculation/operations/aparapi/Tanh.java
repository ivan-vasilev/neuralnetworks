package com.github.neuralnetworks.calculation.operations.aparapi;


/**
 * Tanh activation function
 */
public class Tanh extends AparapiTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public void run()
	{
		int id = getGlobalId() + startIndex;
		elements[id] = tan(elements[id]);
	}
}
