package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Soft Rectified linear unit
 */
public class AparapiSoftReLU extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = -6602713983386107132L;

    @Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiSoftReLUFunction(inputConnections, valuesProvider, targetLayer);
    }

    public static class AparapiSoftReLUFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = 2572354641295173835L;

	public AparapiSoftReLUFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(inputConnections, valuesProvider, targetLayer);
	}

	@Override
	protected void after() {
	    int end = outputStartPosition + getGlobalId() * outputRowStep + miniBatchSize * outputColumnStep;
	    for (int i = outputStartPosition + getGlobalId() * outputRowStep; i < end; i += outputColumnStep) {
		output[i] = log(1 + exp(output[i]));
	    }
	}
    }
}
