package com.github.neuralnetworks.test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * Simple input provider for testing purposes.
 * Training and target data are two dimensional float arrays
 */
public class SimpleInputProvider implements TrainingInputProvider {

    private static final long serialVersionUID = 1L;

    private float[][] input;
    private float[][] target;
    private SimpleTrainingInputData data;
    private int count;
    private int miniBatchSize;
    private int current;

    public SimpleInputProvider(float[][] input, float[][] target, int count, int miniBatchSize) {
	super();

	this.count = count;
	this.miniBatchSize = miniBatchSize;

	data = new SimpleTrainingInputData(null, null);

	if (input != null) {
	    this.input  = input;
	    data.setInput(new Matrix(input[0].length, miniBatchSize));
	}

	if (target != null) {
	    this.target = target;
	    data.setTarget(new Matrix(target[0].length, miniBatchSize));
	}
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
	if (current < count) {
	    for (int i = 0; i < miniBatchSize; i++, current++) {
		if (input != null) {
		    for (int j = 0; j < input[current % input.length].length; j++) {
			data.getInput().set(j, i, input[current % input.length][j]);
		    }
		}

		if (target != null) {
		    for (int j = 0; j < target[current % target.length].length; j++) {
			data.getTarget().set(j, i, target[current % target.length][j]);
		    }
		}
	    }

	    return data;
	}

	return null;
    }

    private static class SimpleTrainingInputData implements TrainingInputData {

	private static final long serialVersionUID = 1L;

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
}
