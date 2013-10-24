package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLUByRows.AparapiReLUByColumns;

public class AparapiReLUConnectionCalculator extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = -6602713983386107132L;

    public AparapiReLUConnectionCalculator() {
	super(new AparapiReLUByRows(), new AparapiReLUByColumns(), null);
    }
}
