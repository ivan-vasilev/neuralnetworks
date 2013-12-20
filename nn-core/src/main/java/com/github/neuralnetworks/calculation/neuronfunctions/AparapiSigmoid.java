package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

public class AparapiSigmoid extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiSigmoid() {
	super(new AparapiSigmoidFunction());
    }

    public static class AparapiSigmoidFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after(int row, int column) {
	    int index = outputBaseIndex(row, column);
	    output[index] = 1 / (1 + exp(-output[index]));
	}
    }
}
