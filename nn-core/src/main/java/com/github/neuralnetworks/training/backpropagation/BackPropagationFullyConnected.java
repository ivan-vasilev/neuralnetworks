package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
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
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, Layer targetLayer) {
	for (Entry<Connections, Integer> e : inputConnections.entrySet()) {
	    SortedMap<GraphConnections, Integer> m = new TreeMap<>();
	    if (Util.isBias(e.getKey().getInputLayer()) && targetLayer != e.getKey().getInputLayer()) {
		m.put((GraphConnections) e.getKey(), miniBatchSize);
		connectionCalculators.put(e.getKey(), new AparapiBackpropagationFullyConnected(m, miniBatchSize, getLearningRate(), getMomentum(), getWeightDecay(), e.getKey().getInputLayer()));
	    } else {
		m.put((GraphConnections) e.getKey(), e.getValue());
		connectionCalculators.put(e.getKey(), new AparapiBackpropagationFullyConnected(m, miniBatchSize, getLearningRate(), getMomentum(), getWeightDecay(), targetLayer));
	    }
	}
    }
}
