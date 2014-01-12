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
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, int inputOutputSamples, Layer targetLayer) {
	return new AparapiSigmoidFunction(inputConnections, inputOutputSamples, targetLayer);
    }

    public static class AparapiSigmoidFunction extends AparapiWeightedSum {
	
	public AparapiSigmoidFunction(SortedMap<GraphConnections, Integer> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}

	private static final long serialVersionUID = -3409078521599849086L;
	
	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] =  1 / (1 + exp(-value));
	}
    }
}
