package com.github.neuralnetworks.neuronfunctions;

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
    protected void outputCalculated(int outputIndex) {
	if (output[outputIndex] < threshold) {
	    output[outputIndex] = 0;
	} else {
	    output[outputIndex] = 1;
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
	protected void outputCalculated(int outputIndex) {
	    if (output[outputIndex] < threshold) {
		output[outputIndex] = 0;
	    } else {
		output[outputIndex] = 1;
	    }
	}
    }
}
