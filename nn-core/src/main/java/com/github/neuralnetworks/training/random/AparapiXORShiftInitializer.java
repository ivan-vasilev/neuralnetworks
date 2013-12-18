package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.util.Environment;

public class AparapiXORShiftInitializer implements RandomInitializer {

    private static final long serialVersionUID = 1L;

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
	XORShift kernel = kernels.get(array.length);
	if (kernel == null) {
	    kernels.put(array.length, kernel = new XORShift(array.length));
	}

	kernel.array = array;
	kernel.start = start;
	kernel.range = range;

	kernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	kernel.execute(array.length);
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
