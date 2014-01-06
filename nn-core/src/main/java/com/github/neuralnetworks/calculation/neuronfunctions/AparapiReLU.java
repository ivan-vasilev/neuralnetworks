package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

/**
 * Rectified linear unit
 */
public class AparapiReLU extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = -6602713983386107132L;

    public AparapiReLU() {
	super(new AparapiReLUFunction());
    }

    public static class AparapiReLUFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = max(0, value);
	}
    }
}
