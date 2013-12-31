package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

/**
 * Tanh activation function
 */
public class AparapiTanh extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiTanh() {
	super(new AparapiTanhFunction());
    }

    public static class AparapiTanhFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = tan(value);
	}
    }
}
