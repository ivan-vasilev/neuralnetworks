package com.github.neuralnetworks.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.AparapiExecutionMode;

/**
 * 
 * Treshold activation function
 * 
 */
public class TresholdFunction implements ActivationFunction {

    private float[] tresholds;
    private Kernel kernel;

    public TresholdFunction(float[] tresholds) {
	super();
	this.tresholds = tresholds;
    }

    @Override
    public void value(Matrix inputOutput) {
	if (kernel == null) {
	    final float[] io = inputOutput.getElements();
	    final int cols = inputOutput.getColumns();
	    final float[] tresholds = this.tresholds;

	    kernel = new Kernel() {
		@Override
		public void run() {
		    int id = getGlobalId();
		    for (int i = 0; i < cols; i++) {
			int idx = id * cols + i;
			if (io[idx] >= tresholds[id]) {
			    io[idx] = 1;
			} else {
			    io[idx] = 0;
			}
		    }
		}
	    };
	}

	kernel.setExecutionMode(AparapiExecutionMode.getInstance().getExecutionMode());

	kernel.execute(inputOutput.getRows());
    }

    public float[] getTresholds() {
	return tresholds;
    }

    public void setTresholds(float[] tresholds) {
	this.tresholds = tresholds;
    }
}
