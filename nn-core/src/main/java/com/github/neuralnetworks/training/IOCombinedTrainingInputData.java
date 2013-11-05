package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * this combines input and output data into the input value
 * 
 */
public class IOCombinedTrainingInputData implements TrainingInputData {

    private TrainingInputData base;

    public IOCombinedTrainingInputData(TrainingInputData base) {
	super();
	this.base = base;
    }

    @Override
    public Matrix getInput() {
	Matrix input = base.getInput();
	Matrix target = base.getTarget();

	if (target != null) {
	    float[] r = new float[input.getElements().length + target.getElements().length];
	    System.arraycopy(input.getElements(), 0, r, 0, input.getElements().length);
	    System.arraycopy(target.getElements(), 0, r, input.getElements().length, target.getElements().length);
	    return new Matrix(r, input.getColumns());
	} else {
	    return input;
	}
    }

    @Override
    public Matrix getTarget() {
	return base.getTarget();
    }
}
