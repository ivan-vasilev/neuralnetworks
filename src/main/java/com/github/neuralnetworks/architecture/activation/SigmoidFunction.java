package com.github.neuralnetworks.architecture.activation;

import org.apache.commons.math3.analysis.function.Sigmoid;

public class SigmoidFunction implements ActivationFunction {

	private Sigmoid sigmoid = new Sigmoid();

	@Override
	public double value(double x) {
		return sigmoid.value(x);
	}
}
