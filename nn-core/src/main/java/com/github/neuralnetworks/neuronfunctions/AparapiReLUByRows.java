package com.github.neuralnetworks.neuronfunctions;


/**
 * Rectified linear unit activation function
 */
public class AparapiReLUByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void outputCalculated(int outputIndex) {
		output[outputIndex] = log(1 + exp(output[outputIndex]));
	}

	public static class AparapiReLUByColumns extends AparapiWeightedSumByColumns {

		private static final long serialVersionUID = 2572354641295173835L;

		@Override
		protected void outputCalculated(int outputIndex) {
			output[outputIndex] = log(1 + exp(output[outputIndex]));
		}
	}
}
