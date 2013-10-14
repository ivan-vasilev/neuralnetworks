package com.github.neuralnetworks.outputerror;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MnistOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Object[] targetOutput) {
	for (int i = 0; i < targetOutput.length; i++, count++) {
	    int val = (int) targetOutput[i];
	    for (int j = 0; j < networkOutput.getRows(); j++) {
		if (networkOutput.get(j, i) > networkOutput.get(val, i)) {
		    totalNetworkError++;
		    break;
		}
	    }
	}
    }

    @Override
    public float getTotalNetworkError() {
	return count > 0 ? totalNetworkError / count : 0;
    }
}
