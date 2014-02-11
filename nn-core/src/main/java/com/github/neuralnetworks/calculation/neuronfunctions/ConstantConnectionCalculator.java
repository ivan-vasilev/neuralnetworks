package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Util;

/**
 * Basic connection calculator that populates the output with a constant (for bias layers)
 */
public class ConstantConnectionCalculator implements ConnectionCalculator {

    private static final long serialVersionUID = -512468674234271624L;

    private float value;

    public ConstantConnectionCalculator() {
	super();
	this.value = 1;
    }

    public ConstantConnectionCalculator(float value) {
	super();
	this.value = value;
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	Util.fillArray(valuesProvider.getValues(targetLayer, connections).getElements(), value);
    }

    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }
}
