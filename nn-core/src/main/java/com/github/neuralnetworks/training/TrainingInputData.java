package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;


/**
 *
 * training input data with target value
 *
 */
public class TrainingInputData {

	private Matrix input;
	private Object[] target;

	public TrainingInputData() {
		super();
	}

	public TrainingInputData(Matrix input) {
		super();
		this.input = input;
	}

	public TrainingInputData(Matrix input, Object[] target) {
		this.input = input;
		this.target = target;
	}

	public Object[] getTarget() {
		return target;
	}

	public void setTarget(Object[] target) {
		this.target = target;
	}

	public Matrix getInput() {
		return input;
	}

	public void setInput(Matrix input) {
		this.input = input;
	}
}
