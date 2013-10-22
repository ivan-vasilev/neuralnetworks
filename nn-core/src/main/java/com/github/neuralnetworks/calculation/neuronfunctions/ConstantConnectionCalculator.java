package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Util;

public class ConstantConnectionCalculator implements ConnectionCalculator {

    private float value;
    
    public ConstantConnectionCalculator(float value) {
	super();
	this.value = value;
    }

    @Override
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer) {
	Util.fillArray(output.getElements(), value);
    }
}
