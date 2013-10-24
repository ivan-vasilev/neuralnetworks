package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * Rectified linear unit activation function
 */
public class AparapiReLUByRows extends AparapiWeightedSumByRows {

    private static final long serialVersionUID = 2572354641295173835L;

    @Override
    protected void after(int row, int column) {
	output[outputBaseIndex(row, column)] = log(1 + exp(output[outputBaseIndex(row, column)]));
    }

    public static class AparapiReLUByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = log(1 + exp(output[outputBaseIndex(row, column)]));
	}
    }
}
