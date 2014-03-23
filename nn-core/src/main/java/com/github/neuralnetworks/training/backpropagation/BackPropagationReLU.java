package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Backpropagation connection calculator for relu units
 */
public class BackPropagationReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationReLU(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	for (Entry<Connections, Integer> e : inputConnections.entrySet()) {
	    SortedMap<GraphConnections, Integer> m = new TreeMap<>();
	    if (Util.isBias(e.getKey().getInputLayer()) && targetLayer != e.getKey().getInputLayer()) {
		m.put((GraphConnections) e.getKey(), miniBatchSize);
		connectionCalculators.put(e.getKey(), new AparapiBackpropReLU(m, miniBatchSize, getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay(), e.getKey().getInputLayer()));
	    } else {
		m.put((GraphConnections) e.getKey(), e.getValue());
		connectionCalculators.put(e.getKey(), new AparapiBackpropReLU(m, miniBatchSize, getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay(), targetLayer));
	    }
	}
    }

    public static class AparapiBackpropReLU extends AparapiBackpropagationFullyConnected {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropReLU(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, Layer targetLayer) {
	    super(inputConnections, miniBatchSize, learningRate, momentum, l1weightDecay, l2weightDecay, targetLayer);
	}
	
	@Override
	protected void calcDerivative() {
	    for (int i = getGlobalId() * miniBatchSize, endIndex = (getGlobalId() + 1) * miniBatchSize; i < endIndex; i++) {
		if (ffActivation[i] <= 0) {
		    output[i] = 0;
		}
	    }
	}
    }
}
