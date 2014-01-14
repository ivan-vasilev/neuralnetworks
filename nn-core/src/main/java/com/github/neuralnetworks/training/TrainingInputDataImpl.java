package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Training input data with target value default implementation
 */
public class TrainingInputDataImpl implements TrainingInputData {

    private Matrix input;
    private Matrix target;

    public TrainingInputDataImpl() {
	super();
    }

    public TrainingInputDataImpl(Matrix input) {
	super();
	this.input = input;
    }

    public TrainingInputDataImpl(Matrix input, Matrix target) {
	this.input = input;
	this.target = target;
    }

    @Override
    public Matrix getInput() {
	return input;
    }

    public void setInput(Matrix input) {
        this.input = input;
    }

    @Override
    public Matrix getTarget() {
	return target;
    }

    public void setTarget(Matrix target) {
        this.target = target;
    }
}
