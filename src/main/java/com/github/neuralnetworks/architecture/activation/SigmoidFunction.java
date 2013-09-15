package com.github.neuralnetworks.architecture.activation;

import org.apache.commons.math3.analysis.function.Sigmoid;

public class SigmoidFunction implements ActivationFunction {

	private Sigmoid sigmoid = new Sigmoid();

	@Override
	public float value(float x) {
		return (float) sigmoid.value(x);
	}
}
