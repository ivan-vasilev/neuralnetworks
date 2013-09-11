package com.github.neuralnetworks.architecture.activation;

/**
 *
 * this function repeats it's input
 *
 */
public class RepeaterFunction implements ActivationFunction {

	@Override
	public double value(double input) {
		return input;
	}
}
