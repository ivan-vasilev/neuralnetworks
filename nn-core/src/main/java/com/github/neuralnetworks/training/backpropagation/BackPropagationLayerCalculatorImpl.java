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

    public BackPropagationLayerCalculatorImpl() {
	super();
    }

    @Override
    public void backpropagate(Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results, Layer layer) {
	BackPropagationConnectionCalculator connectionCalculator = (BackPropagationConnectionCalculator) getConnectionCalculator(layer);
	connectionCalculator.setActivations(activations);
	super.calculate(calculatedLayers, results, layer);
    }


    @Override
    public void addConnectionCalculator(Layer layer, ConnectionCalculator calculator) {
	if (!(calculator instanceof BackPropagationConnectionCalculator)) {
	    throw new IllegalArgumentException("Only BackPropagationConnectionCalculator is allowed");
	}

	super.addConnectionCalculator(layer, calculator);
    }
}