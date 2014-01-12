package com.github.neuralnetworks.samples.mnist;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MnistSingleNeuronOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	for (int i = 0; i < targetOutput.getColumns(); i++, count++) {
	    if (Math.round(networkOutput.get(0, i)) != targetOutput.get(0, i)) {
		totalNetworkError++;
	    }
	}
    }

    @Override
    public float getTotalNetworkError() {
	return count > 0 ? totalNetworkError / count : 0;
    }
}
