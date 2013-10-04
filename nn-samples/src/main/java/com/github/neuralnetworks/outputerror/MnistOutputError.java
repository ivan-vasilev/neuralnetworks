package com.github.neuralnetworks.outputerror;

import com.github.neuralnetworks.calculation.MeanSquaredOutputError;

public class MnistOutputError extends MeanSquaredOutputError {

	private float[] targetArray = new float[10];

	@Override
	protected float[] targetToArray(Object targetOutput) {
		for (int i = 0; i < targetArray.length; i++) {
			targetArray[i] = (int) targetOutput == i ? i : 0;
		}

		return targetArray;
	}
}
