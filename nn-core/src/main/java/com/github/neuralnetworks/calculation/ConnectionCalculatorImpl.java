package com.github.neuralnetworks.calculation;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.ActivationFunction;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;

/**
 * 
 * default implementation for Connection calculator
 *
 */
public class ConnectionCalculatorImpl implements ConnectionCalculator {

    private static final long serialVersionUID = -5405654469496055017L;

    protected ConnectionCalculator forwardInputFunction;
    protected ConnectionCalculator backwardInputFunction;
    protected List<ActivationFunction> activationFunctions;

    public ConnectionCalculatorImpl(ConnectionCalculator forwardInputFunction, ConnectionCalculator backwardInputFunction) {
	super();
	this.forwardInputFunction = forwardInputFunction;
	this.backwardInputFunction = backwardInputFunction;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Matrix> forward = new TreeMap<>();
	SortedMap<Connections, Matrix> backward = new TreeMap<>();
	Map<GraphConnections, Float> bias = new TreeMap<>();

	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    // bias layer scenarios
	    if (c.getOutputLayer() == targetLayer) {
		if (c instanceof GraphConnections && c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		    ConstantConnectionCalculator cc = (ConstantConnectionCalculator) c.getInputLayer().getConnectionCalculator();
		    bias.put((GraphConnections) c, cc.getValue());;
		} else {
		    forward.put(c, input);
		}
	    } else if (c.getInputLayer() == targetLayer) {
		if (c instanceof GraphConnections && c.getOutputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		    ConstantConnectionCalculator cc = (ConstantConnectionCalculator) c.getOutputLayer().getConnectionCalculator();
		    bias.put((GraphConnections) c, cc.getValue());;
		} else {
		    backward.put(c, input);
		}
	    }
	}

	calculateBias(bias, output);
	
	if (forward.size() > 0) {
	    forwardInputFunction.calculate(forward, output, targetLayer);
	}
	
	if (backward.size() > 0) {
	    backwardInputFunction.calculate(backward, output, targetLayer);
	}

	if (activationFunctions != null) {
	    for (ActivationFunction f : activationFunctions) {
		f.value(output);
	    }
	}
    }

    protected void calculateBias(Map<GraphConnections, Float> bias, Matrix output) {
	if (bias.size() > 0) {
	    float[] out = output.getElements();
	    for (int i = 0; i < out.length; i++) {
		for (Entry<GraphConnections, Float> e : bias.entrySet()) {
		    out[i] += e.getKey().getConnectionGraph().getElements()[i / output.getColumns()] * e.getValue();
		}
	    }
	}
    }

    public void addActivationFunction(ActivationFunction activationFunction) {
	if (activationFunctions == null) {
	    activationFunctions = new UniqueList<>();
	}

	activationFunctions.add(activationFunction);
    }

    public void removeActivationFunction(ActivationFunction activationFunction) {
	if (activationFunctions != null) {
	    activationFunctions.remove(activationFunction);
	}
    }
}
