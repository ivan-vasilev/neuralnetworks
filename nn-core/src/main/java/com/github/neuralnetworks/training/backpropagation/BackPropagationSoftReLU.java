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
 * Backpropagation connection calculator for softplus layers
 */
public class BackPropagationSoftReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSoftReLU(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	for (Connections c : inputConnections) {
	    if (Util.isBias(c.getInputLayer()) && targetLayer != c.getInputLayer()) {
		connectionCalculators.put(c, new AparapiBackpropSoftReLU(Arrays.asList(new Connections[] {c}), valuesProvider, c.getInputLayer(), getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    } else {
		connectionCalculators.put(c, new AparapiBackpropSoftReLU(Arrays.asList(new Connections[] {c}), valuesProvider, targetLayer, getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    }
	}
    }

    public static class AparapiBackpropSoftReLU extends AparapiBackpropagationFullyConnected {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropSoftReLU(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer, float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
	    super(inputConnections, valuesProvider, targetLayer, learningRate, momentum, l1weightDecay, l2weightDecay);
	}

	@Override
	protected void calcDerivative() {
	    for (int i = getGlobalId() * miniBatchSize, endIndex = (getGlobalId() + 1) * miniBatchSize; i < endIndex; i++) {
		output[i] = output[i] * (1 / (1 + exp(-ffActivation[i])));
	    }
	}
    }
}
