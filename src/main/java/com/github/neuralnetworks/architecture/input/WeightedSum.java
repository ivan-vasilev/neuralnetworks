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
	public float calc(float[] values, float[] weights) {
		if (values == null || weights == null || values.length != weights.length) {
			throw new IllegalArgumentException();
		}

		float result = 0;
		for (int i = 0; i < values.length; i++) {
			result += values[i] * weights[i];
		}

		return result;
	}

}
