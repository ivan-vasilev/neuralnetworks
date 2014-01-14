package com.github.neuralnetworks.input;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MultipleNeuronsOutputError implements OutputError {

    private float totalNetworkError;
    private int count;

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	for (int i = 0; i < targetOutput.getColumns(); i++, count++) {
	    boolean hasDifferentValues = false;
	    for (int j = 0; j < networkOutput.getRows(); j++) {
		if (networkOutput.get(j, i) != networkOutput.get(0, i)) {
		    hasDifferentValues = true;
		}
	    }
	    
	    if (hasDifferentValues) {
		int val = 0;
		for (int j = 0; j < targetOutput.getRows(); j++) {
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
	    } else {
		totalNetworkError++;
	    }
	}
    }

    @Override
    public float getTotalNetworkError() {
	return count > 0 ? totalNetworkError / count : 0;
    }
}
