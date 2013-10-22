package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.ActivationFunction;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.InputFunction;

/**
 * 
 * default implementation for Connection calculator
 *
 */
public class ConnectionCalculatorImpl implements ConnectionCalculator {

    private InputFunction forwardInputFunction;
    private InputFunction backwardInputFunction;
    private ActivationFunction activationFunction;

    public ConnectionCalculatorImpl(InputFunction forwardInputFunction, InputFunction backwardInputFunction, ActivationFunction activationFunction) {
	super();
	this.forwardInputFunction = forwardInputFunction;
	this.backwardInputFunction = backwardInputFunction;
	this.activationFunction = activationFunction;
    }

    @Override
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer) {
	// bias layer scenarios
	if (connection.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
	    connection.getInputLayer().getConnectionCalculator().calculate(connection, output, input, targetLayer);
	    forwardInputFunction.calculate(connection, input, output);
	} else if (connection.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
	    connection.getOutputLayer().getConnectionCalculator().calculate(connection, output, input, targetLayer);
	    backwardInputFunction.calculate(connection, input, output);
	} else if (targetLayer == connection.getOutputLayer()) {
	    forwardInputFunction.calculate(connection, input, output);
	} else if (targetLayer == connection.getInputLayer()) {
	    backwardInputFunction.calculate(connection, input, output);
	}

	if (activationFunction != null) {
	    activationFunction.value(output);
	}
    }
}
