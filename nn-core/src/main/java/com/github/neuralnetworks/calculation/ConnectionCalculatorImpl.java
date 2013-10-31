package com.github.neuralnetworks.calculation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;
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
	Map<OneToOne, Float> bias = new TreeMap<>();

	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    // bias layer scenarios
	    if (c.getOutputLayer() == targetLayer) {
		if (c instanceof OneToOne && c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		    ConstantConnectionCalculator cc = (ConstantConnectionCalculator) c.getInputLayer().getConnectionCalculator();
		    bias.put((OneToOne) c, cc.getValue());;
		} else {
		    forward.put(c, input);
		}
	    } else if (c.getInputLayer() == targetLayer) {
		if (c instanceof OneToOne && c.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		    ConstantConnectionCalculator cc = (ConstantConnectionCalculator) c.getOutputLayer().getConnectionCalculator();
		    bias.put((OneToOne) c, cc.getValue());;
		} else {
		    backward.put(c, input);
		}
	    }
	}

	if (bias.size() > 0) {
	    float[] out = output.getElements();
	    for (int i = 0; i < out.length; i++) {
		for (Entry<OneToOne, Float> e : bias.entrySet()) {
		    out[i] += e.getKey().getConnectionGraph().getElements()[i / output.getColumns()] * e.getValue();
		}
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
