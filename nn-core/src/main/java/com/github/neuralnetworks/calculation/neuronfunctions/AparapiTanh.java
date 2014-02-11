package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Tanh activation function
 */
public class AparapiTanh extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiTanhFunction(inputConnections, valuesProvider.getColumns(), targetLayer);
    }

    public static class AparapiTanhFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -3409078521599849086L;

	public AparapiTanhFunction(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	    super(inputConnections, miniBatchSize, targetLayer);
	}

	@Override
	protected void after() {
	    int mb = miniBatchSize;
	    int outputId = getGlobalId() * mb;
	    
	    for (int i = 0; i < mb; i++) {
		output[outputId + i] = tan(output[outputId + i]);
	    }
	}
    }
}
