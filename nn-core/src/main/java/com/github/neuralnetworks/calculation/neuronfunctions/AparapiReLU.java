package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

/**
 * Rectified linear unit activation function
 */
public class AparapiReLU extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = -6602713983386107132L;

    public AparapiReLU() {
	super(new AparapiReLUByRows(), new AparapiReLUByColumns());
    }

    public static class AparapiReLUByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = log(1 + exp(output[outputBaseIndex(row, column)]));
	}
    }

    public static class AparapiReLUByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = log(1 + exp(output[outputBaseIndex(row, column)]));
	}
    }
}
