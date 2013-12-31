package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

/**
 * Soft Rectified linear unit
 */
public class AparapiSoftReLU extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = -6602713983386107132L;

    public AparapiSoftReLU() {
	super(new AparapiSoftReLUFunction());
    }

    public static class AparapiSoftReLUFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = log(1 + exp(value));
	}
    }
}
