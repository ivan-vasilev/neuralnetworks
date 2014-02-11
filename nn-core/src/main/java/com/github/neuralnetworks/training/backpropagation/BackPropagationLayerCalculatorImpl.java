package com.github.neuralnetworks.training.backpropagation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorBase;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Util;

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

	Layer currentLayer = nn.getOutputLayer();

	Queue<Layer> layersQueue = new LinkedList<>();
	layersQueue.add(currentLayer);
	Set<Connections> visitedConnections = new HashSet<>();
	List<ConnectionCalculateCandidate> connections = new ArrayList<>();

	while (layersQueue.size() > 0) {
	    Layer l = layersQueue.poll();

	    for (Connections c : l.getConnections(nn)) {
		Layer opposite = Util.getOppositeLayer(c, l);
		if (!visitedConnections.contains(c)) {
		    connections.add(new ConnectionCalculateCandidate(c, opposite));
		    layersQueue.add(opposite);
		    visitedConnections.add(c);
		}
	    }
	}

	calculate(results, connections);
    }

    @Override
    public ConnectionCalculator getConnectionCalculator(Layer layer) {
	ConnectionCalculator cc = super.getConnectionCalculator(layer);
	if (cc != null) {
	    ((BackPropagationConnectionCalculatorImpl) cc).setActivations(activations);
	}

	return cc;
    }

    @Override
    public void addConnectionCalculator(Layer layer, ConnectionCalculator calculator) {
	if (!(calculator instanceof BackPropagationConnectionCalculatorImpl)) {
	    throw new IllegalArgumentException("Only BackPropagationConnectionCalculator is allowed");
	}

	super.addConnectionCalculator(layer, calculator);
    }
}