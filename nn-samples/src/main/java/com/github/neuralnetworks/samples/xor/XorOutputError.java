package com.github.neuralnetworks.samples.xor;

import java.util.Iterator;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.util.Tensor;

public class XorOutputError implements OutputError {

    private static final long serialVersionUID = 1L;

    private float networkError;
    private int size;

    @Override
    public void addItem(Tensor networkOutput, Tensor targetOutput) {
	Iterator<Integer> targetIt = targetOutput.iterator();
	Iterator<Integer> actualIt = networkOutput.iterator();
	while (targetIt.hasNext() && actualIt.hasNext()) {
	    networkError += Math.abs(Math.abs(networkOutput.getElements()[actualIt.next()]) - Math.abs(targetOutput.getElements()[targetIt.next()]));
	}
    }

    @Override
    public float getTotalNetworkError() {
	return size > 0 ? networkError / size : 0;
    }

    @Override
    public int getTotalErrorSamples() {
	return size;
    }

    @Override
    public int getTotalInputSize() {
	return size;
    }

    @Override
    public void reset() {
	networkError = 0;
	size = 0;
    }
}