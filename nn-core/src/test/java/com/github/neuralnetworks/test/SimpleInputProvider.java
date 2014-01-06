package com.github.neuralnetworks.test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * simple input provider for testing purposes
 */
public class SimpleInputProvider implements TrainingInputProvider {

    private Matrix input;
    private Matrix target;
    private int count;
    private int current;

    public SimpleInputProvider(float[][] input, float[][] target, int count) {
	super();

	this.count = count;

	if (input != null) {
	    this.input = new Matrix(input[0].length, input.length);
	    
	    for (int i = 0; i < input.length; i++) {
		for (int j = 0; j < input[i].length; j++) {
		    this.input.set(j, i, input[i][j]);
		}
	    }
	}

	if (target != null) {
	    this.target = new Matrix(target[0].length, target.length);

	    for (int i = 0; i < target.length; i++) {
		for (int j = 0; j < target[i].length; j++) {
		    this.target.set(j, i, target[i][j]);
		}
	    }
	}
    }

    public SimpleInputProvider(Matrix input, int count) {
	super();
	this.input = input;
	this.count = count;
    }

    public SimpleInputProvider(Matrix input, Matrix target, int count) {
	super();
	this.input = input;
	this.target = target;
	this.count = count;
    }

    @Override
    public int getInputSize() {
	return count;
    }

    @Override
    public void reset() {
	current = 0;
    }

    @Override
    public TrainingInputData getNextInput() {
	if (current++ == count) {
	    return null;
	}

	return new SimpleTrainingInputData(input, target);
    }

    private static class SimpleTrainingInputData implements TrainingInputData {

	private Matrix input;
	private Matrix target;

	public SimpleTrainingInputData(Matrix input, Matrix target) {
	    super();
	    this.input = input;
	    this.target = target;
	}

	@Override
	public Matrix getInput() {
	    return input;
	}

	@Override
	public Matrix getTarget() {
	    return target;
	}
    }
}
