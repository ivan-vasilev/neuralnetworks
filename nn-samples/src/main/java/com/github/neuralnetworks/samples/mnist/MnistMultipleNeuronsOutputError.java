package com.github.neuralnetworks.samples.mnist;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MnistMultipleNeuronsOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	for (int i = 0; i < targetOutput.getColumns(); i++, count++) {
	    int val = 0;
	    for (int j = 0; j < 10; j++) {
		if (targetOutput.get(j, i) == 1) {
		    val = j;
		    break;
		}
	    }

	    for (int j = 0; j < networkOutput.getRows(); j++) {
		if (j != val && networkOutput.get(j, i) > networkOutput.get(val, i)) {
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
