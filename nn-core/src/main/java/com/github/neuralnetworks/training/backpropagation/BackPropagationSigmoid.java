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
 * Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationSigmoid extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationSigmoid(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	for (Connections c : inputConnections) {
	    if (Util.isBias(c.getInputLayer()) && targetLayer != c.getInputLayer()) {
		connectionCalculators.put(c, new AparapiBackpropSigmoid(Arrays.asList(new Connections[] {c}), valuesProvider, c.getInputLayer(), getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    } else {
		connectionCalculators.put(c, new AparapiBackpropSigmoid(Arrays.asList(new Connections[] {c}), valuesProvider, targetLayer, getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	    }
	}
    }

    public static class AparapiBackpropSigmoid extends AparapiBackpropagationFullyConnected {

	public AparapiBackpropSigmoid(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer, float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
	    super(inputConnections, valuesProvider, targetLayer, learningRate, momentum, l1weightDecay, l2weightDecay);
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
