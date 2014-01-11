package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Soft Rectified linear unit
 */
public class AparapiSoftReLU extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = -6602713983386107132L;

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	return new AparapiSoftReLUFunction(inputConnections, inputOutputSamples, targetLayer);
    }

    public static class AparapiSoftReLUFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = 2572354641295173835L;

	public AparapiSoftReLUFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}
	
	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = log(1 + exp(value));
	}
    }
}
