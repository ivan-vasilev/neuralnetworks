package com.github.neuralnetworks.outputerror;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MnistOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	for (int i = 0; i < targetOutput.getColumns(); i++, count++) {
	    int val = 0;
	    for (; val < 10; val++) {
		if (targetOutput.get(val, i) == 1) {
		    break;
		}
	    }

	    for (int j = 0; j < networkOutput.getRows(); j++) {
		if (Math.abs(networkOutput.get(j, i)) > Math.abs(networkOutput.get(val, i))) {
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
