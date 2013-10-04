package com.github.neuralnetworks.training;


/**
 *
 * training input data with target value
 *
 */
public class TrainingInputData {

	private float[] input;
	private Object target;

	public TrainingInputData() {
		super();
	}

	public TrainingInputData(float[] input) {
		super();
		this.input = input;
	}

	public TrainingInputData(float[] input, Object target) {
		this.input = input;
		this.target = target;
	}

	public Object getTarget() {
		return target;
	}

	public void setTarget(Object target) {
		this.target = target;
	}

	public float[] getInput() {
		return input;
	}

	public void setInput(float[] input) {
		this.input = input;
	}
}
