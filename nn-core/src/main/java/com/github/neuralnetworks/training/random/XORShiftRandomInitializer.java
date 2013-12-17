package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import com.github.neuralnetworks.util.Environment;

public class XORShiftRandomInitializer implements RandomInitializer {

    protected float mean;
    protected float standardDeviation;
    protected Map<Integer, XORShift> kernels = new HashMap<>();

    public XORShiftRandomInitializer(float mean, float standardDeviation) {
	super();
	this.mean = mean;
	this.standardDeviation = standardDeviation;
    }

    @Override
    public void initialize(float[] array) {
	if (!kernels.containsKey(array.length)) {
	    kernels.put(array.length, new XORShift(array.length));
	}

	XORShift x = kernels.get(array.length);
	x.array = array;
	x.mean = mean;
	x.standardDeviation = standardDeviation;

	x.setExecutionMode(Environment.getInstance().getExecutionMode());
	x.execute(array.length);
    }

    @Override
    public Random getRandom() {
	return null;
    }

    private static class XORShift extends XORShiftKernel {

	private float[] array;
	private float mean;
	private float standardDeviation;

	public XORShift(int maximumRange) {
	    super(maximumRange);
	}

	@Override
	public void run() {
	    array[getGlobalId()] = mean + randomGaussian() * standardDeviation;
	}
    }
}
