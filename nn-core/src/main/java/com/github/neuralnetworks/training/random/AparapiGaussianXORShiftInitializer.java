package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.util.Environment;

public class AparapiGaussianXORShiftInitializer implements RandomInitializer {

    private static final long serialVersionUID = 1L;

    protected float mean;
    protected float standardDeviation;
    protected Map<Integer, XORShift> kernels = new HashMap<>();

    public AparapiGaussianXORShiftInitializer(float mean, float standardDeviation) {
	super();
	this.mean = mean;
	this.standardDeviation = standardDeviation;
    }

    @Override
    public void initialize(float[] array) {
	XORShift kernel = kernels.get(array.length);
	if (kernel == null) {
	    kernels.put(array.length, kernel = new XORShift(array.length));
	}

	kernel.array = array;
	kernel.mean = mean;
	kernel.standardDeviation = standardDeviation;

	kernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	kernel.execute(array.length);
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
