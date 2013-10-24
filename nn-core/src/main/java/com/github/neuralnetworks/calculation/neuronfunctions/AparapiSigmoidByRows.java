package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * 
 * sigmoid function
 * 
 */
public class AparapiSigmoidByRows extends AparapiWeightedSumByRows {

    private static final long serialVersionUID = -3409078521599849086L;

    @Override
    protected void after(int row, int column) {
	output[outputBaseIndex(row, column)] = 1 / (1 + exp(-output[outputBaseIndex(row, column)]));
    }

    public static class AparapiSigmoidByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = 1 / (1 + exp(-output[outputBaseIndex(row, column)]));
	}
    }
}
