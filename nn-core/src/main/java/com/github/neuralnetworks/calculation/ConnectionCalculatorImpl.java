package com.github.neuralnetworks.calculation;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.ActivationFunction;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * 
 * default implementation for Connection calculator
 *
 */
public class ConnectionCalculatorImpl implements ConnectionCalculator {

    private static final long serialVersionUID = -5405654469496055017L;

    private ConnectionCalculator forwardInputFunction;
    private ConnectionCalculator backwardInputFunction;
    private ActivationFunction activationFunction;

    public ConnectionCalculatorImpl(ConnectionCalculator forwardInputFunction, ConnectionCalculator backwardInputFunction, ActivationFunction activationFunction) {
	super();
	this.forwardInputFunction = forwardInputFunction;
	this.backwardInputFunction = backwardInputFunction;
	this.activationFunction = activationFunction;
    }

    @Override
    public void calculate(Map<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Map<Connections, Matrix> forward = new HashMap<>();
	Map<Connections, Matrix> backward = new HashMap<>();
	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    // bias layer scenarios
	    if (c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator || c.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		c.getInputLayer().getConnectionCalculator().calculate(connections, input, targetLayer);
	    }

	    if (c.getInputLayer() == targetLayer) {
		backward.put(c, input);
	    }

	    if (c.getOutputLayer() == targetLayer) {
		forward.put(c, input);
	    }
	}

	if (forward.size() > 0) {
	    forwardInputFunction.calculate(forward, output, targetLayer);
	}
	
	if (backward.size() > 0) {
	    backwardInputFunction.calculate(forward, output, targetLayer);
	}

	if (activationFunction != null) {
	    activationFunction.value(output);
	}
    }
}
