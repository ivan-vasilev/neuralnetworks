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
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, Layer targetLayer) {
	for (Entry<Connections, Integer> e : inputConnections.entrySet()) {
	    SortedMap<GraphConnections, Integer> m = new TreeMap<>();
	    if (Util.isBias(e.getKey().getInputLayer()) && targetLayer != e.getKey().getInputLayer()) {
		m.put((GraphConnections) e.getKey(), miniBatchSize);
		connectionCalculators.put(e.getKey(), new AparapiBackpropSigmoid(m, miniBatchSize, e.getKey().getInputLayer()));
	    } else {
		m.put((GraphConnections) e.getKey(), e.getValue());
		connectionCalculators.put(e.getKey(), new AparapiBackpropSigmoid(m, miniBatchSize, targetLayer));
	    }
	}
    }

    public static class AparapiBackpropSigmoid extends AparapiBackpropagationFullyConnected {

	public AparapiBackpropSigmoid(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	    super(inputConnections, miniBatchSize, targetLayer);
	}

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivative() {
	    float activation = 0;
	    for (int i = getGlobalId() * miniBatchSize, endIndex = (getGlobalId() + 1) * miniBatchSize; i < endIndex; i++) {
		activation = ffActivation[i];
		output[i] = output[i] * activation * (1 - activation);
	    }
	}
    }
}
