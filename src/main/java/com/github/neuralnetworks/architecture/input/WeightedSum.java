package com.github.neuralnetworks.architecture.input;

/**
 * weighted sum input function
 *
 * @author hok
 *
 */
public class WeightedSum implements InputFunction {

	private static final long serialVersionUID = 8650655018964028006L;

	@Override
	public double calc(double[] values, double[] weights) {
		if (values == null || weights == null || values.length != weights.length) {
			throw new IllegalArgumentException();
		}

		double result = 0;
		for (int i = 0; i < values.length; i++) {
			result += values[i] * weights[i];
		}

		return result;
	}

}
