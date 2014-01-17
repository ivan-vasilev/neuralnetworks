package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Sigmoid connection calculator
 */
public class AparapiSigmoid extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, Layer targetLayer) {
	return new AparapiSigmoidFunction(inputConnections, miniBatchSize, targetLayer);
    }

    public static class AparapiSigmoidFunction extends AparapiWeightedSum {

	public AparapiSigmoidFunction(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	    super(inputConnections, miniBatchSize, targetLayer);
	}

	private static final long serialVersionUID = -3409078521599849086L;

	@Override
	protected void after() {
	    int mb = miniBatchSize;
	    int outputId = getGlobalId() * mb;
	    
	    for (int i = 0; i < mb; i++) {
		output[outputId + i] = 1 / (1 + exp(-output[outputId + i]));
	    }
	}
    }
}
