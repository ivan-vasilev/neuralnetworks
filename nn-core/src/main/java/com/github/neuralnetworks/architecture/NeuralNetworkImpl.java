package com.github.neuralnetworks.architecture;

import java.util.Set;

import com.github.neuralnetworks.util.UniqueList;

/**
 * this is the base class for all the neural networks
 */
public abstract class NeuralNetworkImpl implements NeuralNetwork {

    private Set<Layer> layers;

    public NeuralNetworkImpl() {
	super();
	this.layers = new UniqueList<Layer>();
    }

    @Override
    public Layer getInputLayer() {
	hasInboundConnections:
	for (Layer l : layers) {
	    for (Connections c : l.getConnections()) {
		if (isInnerConnection(c) && l == c.getOutputLayer()) {
		    continue hasInboundConnections;
		}
	    }

	    return l;
	}

	return null;
    }

    protected Layer getNoOutboundConnectionsLayer() {
	hasOutboundConnections:
	for (Layer l : layers) {
	    for (Connections c : l.getConnections()) {
		if (isInnerConnection(c) && l == c.getInputLayer()) {
		    continue hasOutboundConnections;
		}
	    }

	    return l;
	}

	return null;
    }

    @Override
    public Set<Connections> getConnections() {
	Set<Connections> result = new UniqueList<>();
	if (layers != null) {
	    for (Layer l : layers) {
		if (l.getConnections() != null) {
		    for (Connections c : l.getConnections()) {
			// both layers of the connection have to be part of the neural network for this connection to be included
			if (isInnerConnection(c)) {
			    result.add(c);
			}
		    }
		}
	    }
	}

	return result;
    }

    protected boolean addLayer(Layer layer) {
	if (layer != null) {
	    if (layers == null) {
		layers = new UniqueList<>();
	    }
	    
	    if (!layers.contains(layer)) {
		layers.add(layer);
		return true;
	    }
	}

	return false;
    }

    protected boolean isInnerConnection(Connections c) {
	if (layers != null) {
	    return layers.contains(c.getInputLayer()) && layers.contains(c.getOutputLayer());
	}

	return false;
    }

    @Override
    public Set<Layer> getLayers() {
	return layers;
    }
}
