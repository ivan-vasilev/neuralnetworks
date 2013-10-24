package com.github.neuralnetworks.calculation.neuronfunctions;

/**
 * 
 * threshold function
 * 
 */
public class AparapiThresholdByRows extends AparapiWeightedSumByRows {

    private static final long serialVersionUID = -3409078521599849086L;

    private float threshold;

    public AparapiThresholdByRows(float threshold) {
	super();
	this.threshold = threshold;
    }

    @Override
    protected void after(int row, int column) {
	if (output[outputBaseIndex(row, column)] < threshold) {
	    output[outputBaseIndex(row, column)] = 0;
	} else {
	    output[outputBaseIndex(row, column)] = 1;
	}
    }

    public static class AparapiThresholdByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -3409078521599849086L;

	private float threshold;

	public AparapiThresholdByColumns(float threshold) {
	    super();
	    this.threshold = threshold;
	}

	@Override
	protected void after(int row, int column) {
	    if (output[outputBaseIndex(row, column)] < threshold) {
		output[outputBaseIndex(row, column)] = 0;
	    } else {
		output[outputBaseIndex(row, column)] = 1;
	    }
	}
    }
}
