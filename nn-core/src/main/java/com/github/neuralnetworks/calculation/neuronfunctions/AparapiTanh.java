package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;

public class AparapiTanh extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiTanh() {
	super(new AparapiTanhByRows(), new AparapiTanhByColumns());
    }

    public static class AparapiTanhByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = tan(output[outputBaseIndex(row, column)]);
	}
    }

    public static class AparapiTanhByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] = tan(output[outputBaseIndex(row, column)]);
	}
    }
}
