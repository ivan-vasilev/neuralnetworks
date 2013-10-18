package com.github.neuralnetworks.neuronfunctions;

/**
 * 
 * sigmoid function
 * 
 */
public class AparapiSigmoidByRows extends AparapiWeightedSumByRows {

    private static final long serialVersionUID = -3409078521599849086L;

    @Override
    protected void outputCalculated(int row, int column) {
	//output[outputIndex] = 1 / (1 + exp(-output[outputIndex]));
	output[outputIndex(row, column)] = 1 / (1 + exp(-output[outputIndex(row, column)]));
    }

    public static class AparapiSigmoidByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void outputCalculated(int row, int column) {
	    //output[outputIndex] = 1 / (1 + exp(-output[outputIndex]));
	    output[outputIndex(row, column)] = 1 / (1 + exp(-output[outputIndex(row, column)]));
	}
    }
}
