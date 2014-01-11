package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation connection calculator for tanh layers
 */
public class BackPropagationTanh extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationTanh(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Matrix> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, int inputOutputSamples, Layer targetLayer) {
	for (Entry<Connections, Matrix> e : inputConnections.entrySet()) {
	    SortedMap<GraphConnections, Matrix> m = new TreeMap<>();
	    m.put((GraphConnections) e.getKey(), e.getValue());
	    connectionCalculators.put(e.getKey(), new AparapiBackpropTanh(m, inputOutputSamples, targetLayer));
	}
    }

    public static class AparapiBackpropTanh extends AparapiBackpropagationFullyConnected {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropTanh(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}

	@Override
	protected void calcDerivativeAfter(float activation, float error, int outputId) {
	    output[outputId] = error * - error * activation * activation;
	}
    }
}
