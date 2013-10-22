package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoidByRows.AparapiSigmoidByColumns;

public class AparapiSigmoidConnectionCalculator extends ConnectionCalculatorImpl {

    public AparapiSigmoidConnectionCalculator() {
	super(new AparapiSigmoidByRows(), new AparapiSigmoidByColumns(), null);
    }
}
