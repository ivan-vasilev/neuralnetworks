package com.github.neuralnetworks.calculation;

import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

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
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Matrix> forward = new TreeMap<>();
	SortedMap<Connections, Matrix> backward = new TreeMap<>();
	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    // bias layer scenarios
	    if (c.getOutputLayer() == targetLayer ||
		c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator ||
		c.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		forward.put(c, input);
	    }

	    if (c.getInputLayer() == targetLayer) {
		backward.put(c, input);
	    }
	}

	if (forward.size() > 0) {
	    forwardInputFunction.calculate(forward, output, targetLayer);
	}
	
	if (backward.size() > 0) {
	    backwardInputFunction.calculate(backward, output, targetLayer);
	}

	if (activationFunction != null) {
	    activationFunction.value(output);
	}
    }
}
