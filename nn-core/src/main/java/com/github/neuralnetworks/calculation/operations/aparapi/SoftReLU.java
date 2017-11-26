package com.github.neuralnetworks.calculation.operations.aparapi;


/**
 * SoftReLU activation function
 */
public class SoftReLU extends AparapiTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public void run()
	{
		int id = getGlobalId() + startIndex;
		elements[id] = log(1 + exp(elements[id]));
	}
}
