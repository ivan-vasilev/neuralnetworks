package com.github.neuralnetworks.input;

import com.github.neuralnetworks.calculation.neuronfunctions.TensorFunction;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.util.Tensor;

/**
 * Scaling input function
 */
public class ScalingInputFunction implements TensorFunction {

    private static final long serialVersionUID = 1L;

    private Float scale;
    private float[] inputScales;

    /**
     * All inputs are scaled according to the scale value
     */
    public ScalingInputFunction(float scale) {
	super();
	this.scale = scale;
    }

    /**
     * All inputs are scaled according to the scale value
     */
    public ScalingInputFunction(TrainingInputProvider input) {
	super();

	input.reset();

	for (int i = 0; i < input.getInputSize(); i++) {
	    input.beforeSample();
	    float[] in = input.getNextInput();
	    if (inputScales == null) {
		inputScales = new float[in.length];
	    }
	    input.afterSample();

	    for (int j = 0; j < in.length; j++) {
		inputScales[j] = Math.abs(in[j]) > inputScales[j] ? Math.abs(in[j]) : inputScales[j];
	    }
	}

	input.reset();
    }

    public float getScale() {
        return scale;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }

    @Override
    public void value(Tensor inputOutput) {
	float[] elements = inputOutput.getElements();
	if (inputScales != null) {
	    int s = inputOutput.getSize() / inputScales.length;
	    for (int i = 0; i < inputOutput.getSize(); i++) {
		elements[inputOutput.getStartOffset() + i] /= inputScales[i / s];
	    }
	} else {
	    for (int i = inputOutput.getStartOffset(); i < inputOutput.getStartOffset() + inputOutput.getSize(); i++) {
		elements[i] /= scale;
	    }
	};
    }
}
