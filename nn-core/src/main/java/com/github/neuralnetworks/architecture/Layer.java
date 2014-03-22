package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * A layer of neurons. Each layer contains a set of connections that link it to other layers.
 * In that sense every neural network is a graph. This is done for maximum versatility.
 * It makes possible the representation of various architectures - committee of machines or parallel networks to be calculated on different GPU devices.
 */
public class Layer implements Serializable {

    private static final long serialVersionUID = 1035633207383317489L;

    /**
     * Set of links to other layers
     */
    private List<Connections> connections;
    
    public Layer() {
	super();
	this.connections = new UniqueList<>();
    }

    /**
     * @param network
     * @return list of connections within the specific neural network
     */
    public List<Connections> getConnections(NeuralNetwork network) {
	return connections.stream().filter(c -> network.getLayers().contains(Util.getOppositeLayer(c, this))).collect(Collectors.toList());
    }

    public List<Connections> getConnections() {
	return connections;
    }

    public void setConnections(List<Connections> connections) {
	this.connections = connections;
    }

    public void addConnection(Connections connection) {
	if (connections == null) {
	    connections = new UniqueList<>();
	}

	connections.add(connection);
    }

    public int getUnitCount(Collection<Connections> connections) {
	int result = 0;
	for (Connections c : connections) {
	    if (c.getInputLayer() == this) {
		if (result == 0) {
		    result = c.getInputUnitCount();
		}

		if (result != c.getInputUnitCount()) {
		    throw new IllegalArgumentException("Some connections require different unit count");
		}
	    } else if (c.getOutputLayer() == this) {
		if (result == 0) {
		    result = c.getOutputUnitCount();
		}

		if (result != c.getOutputUnitCount()) {
		    throw new IllegalArgumentException("Some connections require different unit count");
		}
	    } else {
		throw new IllegalArgumentException("A connection doesn't have the targetLayer as either input or output");
	    }
	}

	return result;
    }
}
