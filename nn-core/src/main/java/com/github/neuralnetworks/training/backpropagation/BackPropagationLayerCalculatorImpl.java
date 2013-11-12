package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;

/**
 * Aparapi implementation of the backpropagation algorithm
 */
public class BackPropagationLayerCalculatorImpl extends LayerCalculatorImpl implements BackPropagationLayerCalculator {

    private static final long serialVersionUID = 1L;

    private BackPropagationConnectionCalculator connectionCalculator;

    public BackPropagationLayerCalculatorImpl(BackPropagationConnectionCalculator connectionCalculator) {
	super();
	this.connectionCalculator = connectionCalculator;
    }

    @Override
    protected ConnectionCalculator getConnectionCalculator(Layer layer) {
	return connectionCalculator;
    }

    @Override
    public void backpropagate(Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results, Layer layer) {
	connectionCalculator.setActivations(activations);
	super.calculate(calculatedLayers, results, layer);
    }
}