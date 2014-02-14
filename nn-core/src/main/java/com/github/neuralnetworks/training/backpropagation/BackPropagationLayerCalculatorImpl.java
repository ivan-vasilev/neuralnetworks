package com.github.neuralnetworks.training.backpropagation;

import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorBase;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Aparapi implementation of the backpropagation algorithm
 */
public class BackPropagationLayerCalculatorImpl extends LayerCalculatorBase implements BackPropagationLayerCalculator {

    private static final long serialVersionUID = 1L;

    private ValuesProvider activations;

    public BackPropagationLayerCalculatorImpl() {
	super();
    }

    @Override
    public void backpropagate(NeuralNetwork nn, Set<Layer> calculatedLayers, ValuesProvider activations, ValuesProvider results) {
	this.activations = activations;

	List<ConnectionCandidate> connections = new BreadthFirstOrderStrategy(nn, nn.getOutputLayer()).order();

	calculate(results, connections, nn);
    }

    @Override
    public ConnectionCalculator getConnectionCalculator(Layer layer) {
	ConnectionCalculator cc = super.getConnectionCalculator(layer);
	if (cc instanceof BackPropagationConnectionCalculatorImpl) {
	    ((BackPropagationConnectionCalculatorImpl) cc).setActivations(activations);
	}

	return cc;
    }
}