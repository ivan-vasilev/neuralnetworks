package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.random.XORShiftKernel;

/**
 * Random noise
 */
public class AparapiNoise extends XORShiftKernel implements TensorFunction {

    private static final long serialVersionUID = 1L;

    private final float corruptionLevel;
    private final int startIndex;
    private float[] inputOutput;
    private final float corruptedValue;

    public AparapiNoise(Tensor inputOutput, int maximumRange, float corruptionLevel, float corruptedValue) {
	super(maximumRange);
	this.inputOutput = inputOutput.getElements();
	this.startIndex = inputOutput.getStartIndex();
	this.corruptionLevel = corruptionLevel;
	this.corruptedValue = corruptedValue;
    }

    @Override
    public void value(Tensor inputOutput) {
	if (inputOutput.getElements() != this.inputOutput) {
	    if (startIndex != inputOutput.getStartIndex()) {
		throw new IllegalArgumentException("Different tensors");
	    }

	    this.inputOutput = inputOutput.getElements();
	}

	execute(inputOutput.getSize());
    }

    @Override
    public void run() {
	int id = getGlobalId();
	if (random01() < corruptionLevel) {
	    inputOutput[startIndex + id] = corruptedValue;
	}
    }
}
