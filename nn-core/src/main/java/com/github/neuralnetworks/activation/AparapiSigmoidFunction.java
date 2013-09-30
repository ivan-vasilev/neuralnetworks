package com.github.neuralnetworks.activation;

import com.amd.aparapi.Kernel;


public class AparapiSigmoidFunction implements ActivationFunction {

	@Override
	public void value(final float[] inputOutput) {
		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
				inputOutput[id] = 1 / (1 + exp(-inputOutput[id]));
			}
		};
		kernel.execute(inputOutput.length);
	}
}
