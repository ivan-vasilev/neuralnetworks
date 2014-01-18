package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.util.Environment;

public class AparapiGaussianXORShiftInitializer implements RandomInitializer {

    private static final long serialVersionUID = 1L;

    protected float mean;
    protected float standardDeviation;
    protected Map<Integer, XORShift> kernels = new HashMap<>();
    
    public AparapiGaussianXORShiftInitializer() {
	super();
	this.mean = 0;
	this.standardDeviation = 1;
    }

    public AparapiGaussianXORShiftInitializer(float mean, float standardDeviation) {
	super();
	this.mean = mean;
	this.standardDeviation = standardDeviation;
    }

    @Override
    public void initialize(float[] array) {
	XORShift kernel = kernels.get(array.length);
	if (kernel == null) {
	    kernels.put(array.length, kernel = new XORShift(array.length, mean, standardDeviation));
	}

	kernel.array = array;

	Environment.getInstance().getExecutionStrategy().execute(kernel, array.length);
    }

    private static class XORShift extends XORShiftKernel {

	private float[] array;
	private final float mean;
	private final float standardDeviation;

	public XORShift(int maximumRange, float mean, float standardDeviation) {
	    super(maximumRange);
	    this.mean = mean;
	    this.standardDeviation = standardDeviation;
	}

	@Override
	public void run() {
	    array[getGlobalId()] = mean + randomGaussian() * standardDeviation;
	}
    }
}
