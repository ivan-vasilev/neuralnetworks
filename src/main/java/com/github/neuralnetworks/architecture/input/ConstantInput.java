package com.github.neuralnetworks.architecture.input;

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
	public float calc(float[] values, float[] weights) {
		return output;
	}
}
