package com.github.neuralnetworks.neuronfunctions;

/**
 * Rectified linear unit activation function
 */
public class AparapiReLUByRows extends AparapiWeightedSumByRows {

    private static final long serialVersionUID = 2572354641295173835L;

    @Override
    protected void outputCalculated(int row, int column) {
	//output[outputIndex] = log(1 + exp(output[outputIndex]));
	output[outputIndex(row, column)] = log(1 + exp(output[outputIndex(row, column)]));
    }

    public static class AparapiReLUByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void outputCalculated(int row, int column) {
	    //output[outputIndex] = log(1 + exp(output[outputIndex]));
	    output[outputIndex(row, column)] = log(1 + exp(output[outputIndex(row, column)]));
	}
    }
}
