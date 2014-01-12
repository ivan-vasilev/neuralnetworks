package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Rectified linear unit
 */
public class AparapiReLU extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = -6602713983386107132L;

    public AparapiReLU() {
	super();
    }

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, int inputOutputSamples, Layer targetLayer) {
	return new AparapiReLUFunction(inputConnections, inputOutputSamples, targetLayer);
    }

    public static class AparapiReLUFunction extends AparapiWeightedSum {
	
	public AparapiReLUFunction(SortedMap<GraphConnections, Integer> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}

	private static final long serialVersionUID = 2572354641295173835L;
	
	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = max(0, value);
	}
    }
}
