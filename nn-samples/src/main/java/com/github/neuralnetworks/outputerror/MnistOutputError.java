package com.github.neuralnetworks.outputerror;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MnistOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Object[] targetOutput) {
	count++;

	for (int i = 0; i < targetOutput.length; i++, count++) {
	    int maxIndex = 0;
	    int max = 0;
	    for (int j = 1; j < networkOutput.getRows(); j++) {
		if (networkOutput.getElements()[j * targetOutput.length + i] > networkOutput.getElements()[maxIndex]) {
		    maxIndex = j * targetOutput.length + i;
		    max = j;
		}
	    }

	    totalNetworkError += (max != (int) targetOutput[maxIndex] ? 1 : 0);
	}
    }

    @Override
    public float getTotalNetworkError() {
	return count > 0 ? totalNetworkError / count : 0;
    }
}
