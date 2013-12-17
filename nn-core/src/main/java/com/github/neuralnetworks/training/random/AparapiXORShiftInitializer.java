package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.util.Environment;

public class AparapiXORShiftInitializer implements RandomInitializer {
    protected float start;
    protected float range;
    protected Map<Integer, XORShift> kernels = new HashMap<>();

    public AparapiXORShiftInitializer(float start, float end) {
	super();
	this.start = start;
	this.range = end - start;
    }

    @Override
    public void initialize(float[] array) {
	if (!kernels.containsKey(array.length)) {
	    kernels.put(array.length, new XORShift(array.length));
	}

	XORShift x = kernels.get(array.length);
	x.array = array;
	x.start = start;
	x.range = range;

	x.setExecutionMode(Environment.getInstance().getExecutionMode());
	x.execute(array.length);
    }

    private static class XORShift extends XORShiftKernel {

	private float[] array;
	private float start;
	private float range;

	public XORShift(int maximumRange) {
	    super(maximumRange);
	}

	@Override
	public void run() {
	    array[getGlobalId()] = start + random01() * range;
	}
    }
}
