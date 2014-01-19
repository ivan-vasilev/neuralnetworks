package com.github.neuralnetworks.training.backpropagation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorBase;
import com.github.neuralnetworks.util.Util;

/**
 * Aparapi implementation of the backpropagation algorithm
 */
public class BackPropagationLayerCalculatorImpl extends LayerCalculatorBase implements BackPropagationLayerCalculator {

    private static final long serialVersionUID = 1L;

    private Map<Layer, Matrix> activations;

    public BackPropagationLayerCalculatorImpl() {
	super();
    }

    @Override
    public void backpropagate(NeuralNetwork nn, Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results) {
	this.activations = activations;

	Layer currentLayer = nn.getOutputLayer();

	Queue<Layer> layersQueue = new LinkedList<>();
	layersQueue.add(currentLayer);
	Set<Connections> visitedConnections = new HashSet<>();
	List<ConnectionCalculateCandidate> connections = new ArrayList<>();

	while (layersQueue.size() > 0) {
	    Layer l = layersQueue.poll();
	    if (l instanceof BiasLayer && !activations.containsKey(l)) {
		Matrix m = getLayerResult(activations, l);
		Util.fillArray(m.getElements(), 1);
		activations.put(l, m);
	    }

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