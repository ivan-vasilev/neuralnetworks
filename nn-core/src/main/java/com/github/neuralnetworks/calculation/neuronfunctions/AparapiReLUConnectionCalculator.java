package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLUByRows.AparapiReLUByColumns;

public class AparapiReLUConnectionCalculator extends ConnectionCalculatorImpl {

    public AparapiReLUConnectionCalculator() {
	super(new AparapiReLUByRows(), new AparapiReLUByColumns(), null);
    }
}
