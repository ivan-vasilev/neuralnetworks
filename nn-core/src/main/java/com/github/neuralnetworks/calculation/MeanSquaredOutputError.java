package com.github.neuralnetworks.calculation;

public class MeanSquaredOutputError extends OutputError {

	@Override
	protected void updateTotalNetworkError(float[] delta) {
		double errorSum = 0;
		for (double d : delta) {
			errorSum += (d * d);
		}

		this.totalNetworkError += errorSum / (2 * delta.length);
	}
}
