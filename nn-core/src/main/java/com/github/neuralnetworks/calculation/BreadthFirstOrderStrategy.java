package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Util;

/**
 * Breadth first order strategy
 */
public class BreadthFirstOrderStrategy implements LayerOrderStrategy {

    private static final long serialVersionUID = 1L;

    private NeuralNetwork neuralNetwork;
    private Layer startLayer;

    public BreadthFirstOrderStrategy(NeuralNetwork neuralNetwork, Layer startLayer) {
	super();
	this.neuralNetwork = neuralNetwork;
	this.startLayer = startLayer;
    }

    @Override
    public List<ConnectionCandidate> order() {
	List<ConnectionCandidate> result = new ArrayList<>();

	Layer currentLayer = startLayer;

	Queue<Layer> layersQueue = new LinkedList<>();
	layersQueue.add(currentLayer);
	Set<Connections> visitedConnections = new HashSet<>();

	while (layersQueue.size() > 0) {
	    Layer l = layersQueue.poll();

	    l.getConnections(neuralNetwork).stream().filter(c -> !visitedConnections.contains(c)).forEach(c -> {
		Layer opposite = Util.getOppositeLayer(c, l);
		result.add(new ConnectionCandidate(c, opposite));
		layersQueue.add(opposite);
		visitedConnections.add(c);
	    });
	}

	return result;
    }

    public NeuralNetwork getNeuralNetwork() {
	return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
	this.neuralNetwork = neuralNetwork;
    }

    public Layer getStartLayer() {
	return startLayer;
    }

    public void setStartLayer(Layer startLayer) {
	this.startLayer = startLayer;
    }
}
