package com.github.neuralnetworks.input;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Scaling input modifier
 */
public class ScalingInputModifier implements InputModifier {

    private float scale;

    public ScalingInputModifier(float scale) {
	super();
	this.scale = scale;
    }

    @Override
    public Matrix modify(Matrix input) {
	float[] elements = input.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] /= scale;
	}

	return input;
    }

    public float getScale() {
        return scale;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }
}
