package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Backpropagation connection calculator for fully connected input layers
 */
public class BackPropagationFullyConnected extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationFullyConnected(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	for (Connections c : inputConnections) {
	    if (Util.isBias(c.getInputLayer()) && targetLayer != c.getInputLayer()) {
		connectionCalculators.put(c, new AparapiBackpropagationFullyConnected(Arrays.asList(new Connections[] {c}), valuesProvider, c.getInputLayer(), getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    } else {
		connectionCalculators.put(c, new AparapiBackpropagationFullyConnected(Arrays.asList(new Connections[] {c}), valuesProvider, targetLayer, getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    }
	}
    }
}
