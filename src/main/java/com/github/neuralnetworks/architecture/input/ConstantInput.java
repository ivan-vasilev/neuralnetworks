package com.github.neuralnetworks.architecture.input;

/**
 * this input function always returns constant value
 */
public class ConstantInput implements InputFunction {

	private static final long serialVersionUID = 2792702356148670315L;

	private double output;

	public ConstantInput(double output) {
		super();
		this.output = output;
	}

	public ConstantInput() {
		super();
		this.output = 1;
	}

	@Override
	public double calc(double[] values, double[] weights) {
		return output;
	}
}
