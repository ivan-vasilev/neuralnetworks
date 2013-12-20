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
	protected void after(int row, int column) {
	    int index = outputBaseIndex(row, column);
	    output[index] = tan(output[index]);
	}
    }
}
