package com.github.neuralnetworks.samples.xor;

import java.util.Iterator;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.util.Tensor;

public class XorOutputError implements OutputError {

    private static final long serialVersionUID = 1L;

    private float networkError;
    private int size;
    private int errorSamples;

    @Override
    public void addItem(Tensor networkOutput, Tensor targetOutput) {
	Iterator<Integer> targetIt = targetOutput.iterator();
	Iterator<Integer> actualIt = networkOutput.iterator();
	size += targetOutput.getDimensions()[targetOutput.getDimensions().length - 1];
	float error = 0;
	while (targetIt.hasNext() && actualIt.hasNext()) {
	    error += Math.abs(Math.abs(networkOutput.getElements()[actualIt.next()]) - Math.abs(targetOutput.getElements()[targetIt.next()]));
	}
	networkError += error;
	if (error / targetOutput.getDimensions()[targetOutput.getDimensions().length - 1] > 0.5) {
	    errorSamples += targetOutput.getDimensions()[targetOutput.getDimensions().length - 1];
	}
    }

    @Override
    public float getTotalNetworkError() {
	return size > 0 ? networkError / size : 0;
    }

    @Override
    public int getTotalErrorSamples() {
	return errorSamples;
    }

    @Override
    public int getTotalInputSize() {
	return size;
    }

    @Override
    public void reset() {
	networkError = 0;
	errorSamples = 0;
	size = 0;
    }
}