package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Rectified linear unit
 */
public class AparapiReLU extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = -6602713983386107132L;

    @Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiReLUFunction(inputConnections, valuesProvider, targetLayer);
    }

    public static class AparapiReLUFunction extends AparapiWeightedSum {

	public AparapiReLUFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(inputConnections, valuesProvider, targetLayer);
	}

	private static final long serialVersionUID = 2572354641295173835L;

	@Override
	protected void after() {
	    int end = outputStartPosition + getGlobalId() * outputRowStep + miniBatchSize * outputColumnStep;
	    for (int i = outputStartPosition + getGlobalId() * outputRowStep; i < end; i += outputColumnStep) {
		output[i] = max(0, output[i]);
	    }
	}
    }
}
