package com.github.neuralnetworks.architecture;

import java.util.Set;

import com.github.neuralnetworks.util.UniqueList;

/**
 * this is the base class for all the neural networks
 */
public class NeuralNetworkImpl implements NeuralNetwork {

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

    @Override
    public Layer getDataOutputLayer() {
	return getOutputLayer();
    }

    @Override
    public Set<Connections> getConnections() {
	Set<Connections> result = new UniqueList<>();
	for (Layer l : layers) {
	    result.addAll(l.getConnections());
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

    @Override
    public Set<Layer> getLayers() {
	return layers;
    }
}
