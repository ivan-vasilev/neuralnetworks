package com.github.neuralnetworks.input;

import java.util.Arrays;

import com.github.neuralnetworks.architecture.IConnections;

/**
 * this input function always returns constant value
 */
public class ConstantInput implements InputFunction {

	private static final long serialVersionUID = 2792702356148670315L;

	private float output;

	public ConstantInput(float output) {
		super();
		this.output = output;
	}

	public ConstantInput() {
		super();
		this.output = 1;
	}

	@Override
	public void calculateForward(IConnections graph, float[] inputValues, float[] result) {
		Arrays.fill(result, output);
	}

	@Override
	public void calculateBackward(IConnections graph, float[] inputValues, float[] result) {
		Arrays.fill(result, output);
	}
}