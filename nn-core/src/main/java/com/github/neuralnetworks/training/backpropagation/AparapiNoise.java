package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.training.random.XORShiftKernel;
import com.github.neuralnetworks.util.Environment;

/**
 * Aparapi implementation of input data random uniform corruption. Corrupted values are set to 0.
 */
public class AparapiNoise implements InputCorruptor {

    private static final long serialVersionUID = 1L;

    private float corruptionLevel;
    protected Map<Integer, XORShiftNoise> kernels = new HashMap<>();

    public AparapiNoise(float corruptionLevel) {
	super();
	this.corruptionLevel = corruptionLevel;
    }

    @Override
    public void corrupt(float[] values) {
	XORShiftNoise x = kernels.get(values.length);
	if (x == null) {
	    kernels.put(values.length, x = new XORShiftNoise(values.length));
	}

	x.values = values;

	Environment.getInstance().getExecutionStrategy().execute(x, values.length);
    }

    public float getCorruptionLevel() {
	return corruptionLevel;
    }

    public void setCorruptionLevel(float corruptionLevel) {
	this.corruptionLevel = corruptionLevel;
    }

    private static class XORShiftNoise extends XORShiftKernel {

	private float[] values;
	private float corruptionLevel;

	public XORShiftNoise(int maximumRange) {
	    super(maximumRange);
	}

	@Override
	public void run() {
	    if (random01() < corruptionLevel) {
		values[getGlobalId()] = 0;
	    }
	}
    }
}
