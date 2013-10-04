package com.github.neuralnetworks.calculation;

/**
 *
 * class for calculating output error
 *
 */
public abstract class OutputError {

	protected int iterations;
	protected float totalNetworkError;

	public float[] delta(float[] networkOutput, Object targetOutput) {
		float[] delta = new float[networkOutput.length];

		float[] targetArray = targetToArray(targetOutput);
		for (int i = 0; i < delta.length; i++) {
			delta[i] = targetArray[i] - networkOutput[i];
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
	protected abstract float[] targetToArray(Object targetOutput);
}
