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
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] =  1 / (1 + exp(-value));
	}
    }
}
