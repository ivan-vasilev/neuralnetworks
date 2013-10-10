package com.github.neuralnetworks.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Util;

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

	public float getOutput() {
		return output;
	}

	public void setOutput(float output) {
		this.output = output;
	}

	@Override
	public void calculate(Connections graph, Matrix inputValues, Matrix result) {
		Util.fillArray(result.getElements(), output);
	}
}