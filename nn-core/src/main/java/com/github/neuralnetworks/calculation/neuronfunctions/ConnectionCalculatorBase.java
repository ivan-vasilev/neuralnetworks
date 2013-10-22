package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

public class ConnectionCalculatorBase implements ConnectionCalculator {

    @Override
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer) {
	// calculates bias layers first
	if (connection.getOutputLayer() != targetLayer && connection.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
	    connection.getOutputLayer().getConnectionCalculator().calculate(connection, output, input, targetLayer);
	} else if (connection.getInputLayer() != targetLayer && connection.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
	    connection.getInputLayer().getConnectionCalculator().calculate(connection, output, input, targetLayer);
	}
    }
}
