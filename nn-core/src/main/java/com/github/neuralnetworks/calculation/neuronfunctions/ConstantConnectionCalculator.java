package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
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
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Util.fillArray(output.getElements(), value);
    }

    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }
}
