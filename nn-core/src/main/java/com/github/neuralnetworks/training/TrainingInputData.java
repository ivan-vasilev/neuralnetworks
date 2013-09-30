package com.github.neuralnetworks.training;

import java.io.Serializable;

/**
 *
 * training input data with target value
 *
 */
public class TrainingInputData implements Serializable {

	private static final long serialVersionUID = 4518309418687563264L;

	private float[] input;
	private float[] target;

	public TrainingInputData() {
		super();
	}

	public TrainingInputData(float[] input) {
		super();
		this.input = input;
	}

	public TrainingInputData(float[] input, float[] target) {
		this.input = input;
		this.target = target;
	}

	public float[] getTarget() {
		return target;
	}

	public void setTarget(float[] target) {
		this.target = target;
	}

	public float[] getInput() {
		return input;
	}

	public void setInput(float[] input) {
		this.input = input;
	}
}
