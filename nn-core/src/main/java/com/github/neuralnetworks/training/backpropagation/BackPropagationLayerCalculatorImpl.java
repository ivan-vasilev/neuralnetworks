package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
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
    public void backpropagate(NeuralNetwork nn, Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results) {
	Layer currentLayer = nn.getInputLayer();

	while (currentLayer != null) {
	    BackPropagationConnectionCalculatorImpl connectionCalculator = (BackPropagationConnectionCalculatorImpl) getConnectionCalculator(currentLayer);
	    connectionCalculator.setActivations(activations);
	    super.calculate(calculatedLayers, results, currentLayer);

	    currentLayer = null;
	    for (Layer l : nn.getLayers()) {
		if (!calculatedLayers.contains(l)) {
		    currentLayer = l;
		    break;
		}
	    }
	}
    }

    @Override
    public void addConnectionCalculator(Layer layer, ConnectionCalculator calculator) {
	if (!(calculator instanceof BackPropagationConnectionCalculatorImpl)) {
	    throw new IllegalArgumentException("Only BackPropagationConnectionCalculator is allowed");
	}

	super.addConnectionCalculator(layer, calculator);
    }
}