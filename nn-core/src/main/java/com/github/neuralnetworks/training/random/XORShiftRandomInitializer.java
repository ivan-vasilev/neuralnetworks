package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import com.github.neuralnetworks.util.Environment;

public class XORShiftRandomInitializer implements RandomInitializer {

    protected float start;
    protected float end;
    protected Map<Integer, XORShift> kernels = new HashMap<>();

    public XORShiftRandomInitializer(float start, float end) {
	super();
	this.start = start;
	this.end = end;
    }

    @Override
    public void initialize(float[] array) {
	if (!kernels.containsKey(array.length)) {
	    kernels.put(array.length, new XORShift(array.length));
	}

	XORShift x = kernels.get(array.length);
	x.array = array;
	x.range = end - start;

	x.setExecutionMode(Environment.getInstance().getExecutionMode());
	x.execute(array.length);
    }

    @Override
    public Random getRandom() {
	return null;
    }

    public float getStart() {
	return start;
    }

    public void setStart(float start) {
	this.start = start;
    }

    public float getEnd() {
	return end;
    }

    public void setEnd(float end) {
	this.end = end;
    }

    private static class XORShift extends XORShiftKernel {

	private float[] array;
	private float range;

	public XORShift(int maximumRange) {
	    super(maximumRange);
	}

	@Override
	public void run() {
	    array[getGlobalId()] = randomGaussian() * range;
	}
    }
}
