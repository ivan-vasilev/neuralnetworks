package com.github.neuralnetworks.samples.xor;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.util.Matrix;

public class XorOutputError implements OutputError {

    private static final long serialVersionUID = 1L;

    private float networkError;
    private int size;

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	for (int i = 0; i < targetOutput.getColumns(); i++, size++) {
	    networkError += Math.abs(Math.abs(networkOutput.get(0, i)) - Math.abs(targetOutput.get(0, i)));
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