package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid.AparapiSigmoidFunction;

/**
 * Tanh activation function
 */
public class AparapiTanh extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	return new AparapiSigmoidFunction(inputConnections, inputOutputSamples, targetLayer);
    }

    public static class AparapiTanhFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -3409078521599849086L;
	
	public AparapiTanhFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	    super(inputConnections, inputOutputSamples, targetLayer);
	}

	@Override
	protected void after(float value, int row, int column) {
	    output[outputIndex(row, column)] = tan(value);
	}
    }
}
