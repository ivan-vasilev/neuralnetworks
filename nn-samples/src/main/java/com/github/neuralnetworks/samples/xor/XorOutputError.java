package com.github.neuralnetworks.samples.xor;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class XorOutputError implements OutputError {

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
}