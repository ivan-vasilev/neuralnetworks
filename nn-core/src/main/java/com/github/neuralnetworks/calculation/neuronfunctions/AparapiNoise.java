package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.training.random.XORShiftKernel;
import com.github.neuralnetworks.util.Tensor;

public class AparapiNoise extends XORShiftKernel implements TensorFunction {

    private static final long serialVersionUID = 1L;

    private final float corruptionLevel;
    private float[] inputOutput;

    public AparapiNoise(int maximumRange, float corruptionLevel) {
	super(maximumRange);
	this.corruptionLevel = corruptionLevel;
    }

    @Override
    public void value(Tensor inputOutput) {
	this.inputOutput = inputOutput.getElements();
	execute(this.inputOutput.length);
    }

    @Override
    public void run() {
	int id = getGlobalId();
	if (random01() < corruptionLevel) {
	    inputOutput[id] = 0;
	}
    }
}
