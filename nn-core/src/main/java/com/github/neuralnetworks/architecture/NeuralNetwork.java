package com.github.neuralnetworks.architecture;

import java.util.Set;

import com.github.neuralnetworks.util.UniqueList;

/**
 * this is the base class for all the neural networks
 */
public class NeuralNetwork implements InputOutputLayers {

    private Set<Layer> layers;

    public NeuralNetwork() {
	super();
	this.layers = new UniqueList<Layer>();
    }

    @Override
    public Layer getInputLayer() {
	hasInboundConnections:
	for (Layer l : layers) {
	    for (Connections c : l.getConnections()) {
		if (l == c.getOutputLayer()) {
		    continue hasInboundConnections;
		}
	    }

	    return l;
	}

	return null;
    }

    @Override
    public Layer getOutputLayer() {
	hasOutboundConnections:
	for (Layer l : layers) {
	    for (Connections c : l.getConnections()) {
		if (l == c.getInputLayer()) {
		    continue hasOutboundConnections;
		}
	    }

	    return l;
	}

	return null;
    }

    public Set<Connections> getConnections() {
	Set<Connections> result = new UniqueList<>();
	for (Layer l : layers) {
	    result.addAll(l.getConnections());
	}

	return result;
    }

    public void addConnection(Connections c) {
	if (!layers.contains(c.getInputLayer())) {
	    layers.add(c.getInputLayer());
	}

	if (!layers.contains(c.getOutputLayer())) {
	    layers.add(c.getOutputLayer());
	}
    }

    public Set<Layer> getLayers() {
	return layers;
    }
}
