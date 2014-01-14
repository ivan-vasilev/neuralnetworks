package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, int inputOutputSamples, Layer targetLayer) {
	for (Entry<Connections, Integer> e : inputConnections.entrySet()) {
	    SortedMap<GraphConnections, Integer> m = new TreeMap<>();
	    if (e.getKey().getInputLayer() instanceof BiasLayer && targetLayer != e.getKey().getInputLayer()) {
		m.put((GraphConnections) e.getKey(), inputOutputSamples);
		connectionCalculators.put(e.getKey(), new AparapiBackpropSigmoid(m, e.getValue(), e.getKey().getInputLayer()));
	    } else {
		m.put((GraphConnections) e.getKey(), e.getValue());
		connectionCalculators.put(e.getKey(), new AparapiBackpropSigmoid(m, inputOutputSamples, targetLayer));
	    }
	}
    }

    public static class AparapiBackpropSigmoid extends AparapiBackpropagationFullyConnected {

	public AparapiBackpropSigmoid(SortedMap<GraphConnections, Integer> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}

	private static final long serialVersionUID = -3580345016542506932L;

	@Override
	protected void calcDerivativeAfter(float activation, float error, int outputId) {
	    output[outputId] = error * activation * (1 - activation);
	}
    }
}
