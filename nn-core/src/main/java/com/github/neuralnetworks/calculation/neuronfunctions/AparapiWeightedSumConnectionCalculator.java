package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Simple weighted sum connection calculator
 */
public class AparapiWeightedSumConnectionCalculator extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, Layer targetLayer) {
	return new AparapiWeightedSum(inputConnections, miniBatchSize, targetLayer);
    }
}
