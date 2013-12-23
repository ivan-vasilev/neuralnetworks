package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Set;

import com.github.neuralnetworks.util.UniqueList;

/**
 * A layer of neurons. Each layer contains a set of connections that link it to other layers.
 * In that sense every neural network is a graph. This is done for maximum versatility.
 * It makes possible the representation of various architectures - committee of machines or parallel networks to be calculated on different GPU devices.
 */
public class Layer implements Serializable {

    private static final long serialVersionUID = 1035633207383317489L;

    /**
     * Number of neurons in the layer
     */
    private int neuronCount;

    /**
     * Set of links to other layers
     */
    private Set<Connections> connections;
    
    public Layer() {
	super();
	this.connections = new UniqueList<>();
    }

    public Layer(int neuronCount) {
	super();
	this.neuronCount = neuronCount;
	this.connections = new UniqueList<>();
    }

    public int getNeuronCount() {
	return neuronCount;
    }

    public void setNeuronCount(int neuronCount) {
	this.neuronCount = neuronCount;
    }

//    public ConnectionCalculator getConnectionCalculator() {
//	return connectionCalculator;
//    }
//
//    public void setConnectionCalculator(ConnectionCalculator connectionCalculator) {
//	this.connectionCalculator = connectionCalculator;
//    }

    public Set<Connections> getConnections() {
	return connections;
    }

    public void setConnections(Set<Connections> connections) {
	this.connections = connections;
    }

    public void addConnection(Connections connection) {
	if (connections == null) {
	    connections = new UniqueList<>();
	}

	connections.add(connection);
    }
}
