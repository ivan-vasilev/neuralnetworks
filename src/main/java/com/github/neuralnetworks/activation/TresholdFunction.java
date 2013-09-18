package com.github.neuralnetworks.activation;

import com.amd.aparapi.Kernel;

/**
 *
 * Treshold activation function
 *
 */
public class TresholdFunction implements ActivationFunction {

	private float[] tresholds;

	public TresholdFunction(float[] tresholds) {
		super();
		this.tresholds = tresholds;
	}

	@Override
	public void value(final float[] inputOutput) {
		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
				if (inputOutput[id] >= tresholds[id]) {
					inputOutput[id] = 1;
				} else {
					inputOutput[id] = 0;
				}
			}
		};
		kernel.execute(inputOutput.length);
	}

	public float[] getTresholds() {
		return tresholds;
	}

	public void setTresholds(float[] tresholds) {
		this.tresholds = tresholds;
	}
}
