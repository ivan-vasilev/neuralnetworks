package com.github.neuralnetworks.calculation;

/**
 *
 * class for calculating output error
 *
 */
public abstract class OutputError {

	protected int iterations;
	protected float totalNetworkError;

	public float[] delta(float[] networkOutput, float[] targetOutput) {
		if (networkOutput.length != targetOutput.length) {
			throw new IllegalArgumentException("network output and target output must be with the same dimensionality");
		}

		float[] delta = new float[targetOutput.length];

		for (int i = 0; i < delta.length; i++) {
			delta[i] = targetOutput[i] - networkOutput[i];
		}

		updateTotalNetworkError(delta);

		return delta;
	}

	public float getTotalNetworkError() {
		return totalNetworkError;
	}

	public void setTotalNetworkError(float totalNetworkError) {
		this.totalNetworkError = totalNetworkError;
	}

	protected abstract void updateTotalNetworkError(float[] delta);
}
