package com.github.neuralnetworks.input;

import com.github.neuralnetworks.calculation.neuronfunctions.MatrixFunction;
import com.github.neuralnetworks.util.Matrix;

/**
 * Scaling input function
 */
public class ScalingInputFunction implements MatrixFunction {

    private static final long serialVersionUID = 1L;

    private float scale;

    public ScalingInputFunction(float scale) {
	super();
	this.scale = scale;
    }

    public float getScale() {
        return scale;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }

    @Override
    public void value(Matrix inputOutput) {
	float[] elements = inputOutput.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] /= scale;
	}
    }
}
