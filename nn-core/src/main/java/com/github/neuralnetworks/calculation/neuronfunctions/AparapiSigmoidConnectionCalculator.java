package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoidByRows.AparapiSigmoidByColumns;

public class AparapiSigmoidConnectionCalculator extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiSigmoidConnectionCalculator() {
	super(new AparapiSigmoidByRows(), new AparapiSigmoidByColumns(), null);
    }
}
