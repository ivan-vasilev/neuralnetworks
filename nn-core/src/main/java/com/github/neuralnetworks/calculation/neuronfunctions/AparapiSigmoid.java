package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Sigmoid connection calculator
 */
public class AparapiSigmoid extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiSigmoidFunction(inputConnections, valuesProvider, targetLayer);
    }

    public static class AparapiSigmoidFunction extends AparapiWeightedSum {

	public AparapiSigmoidFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(inputConnections, valuesProvider, targetLayer);
	}

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after() {
	    int end = outputStartPosition + getGlobalId() * outputRowStep + miniBatchSize * outputColumnStep;
	    for (int i = outputStartPosition + getGlobalId() * outputRowStep; i < end; i += outputColumnStep) {
		output[i] = 1 / (1 + exp(-output[i]));
	    }
	}
    }
}
