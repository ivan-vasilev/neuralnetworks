package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Set;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;

/**
 * 
 * represents a layer of neurons
 * 
 */
public class Layer implements Serializable {

    private static final long serialVersionUID = 1035633207383317489L;

    private int neuronCount;
    private ConnectionCalculator connectionCalculator;
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

    public Layer(ConnectionCalculator connectionCalculator) {
	super();
	this.connectionCalculator = connectionCalculator;
	this.connections = new UniqueList<>();
    }

    public Layer(int neuronCount, ConnectionCalculator connectionCalculator) {
	super();
	this.neuronCount = neuronCount;
	this.connectionCalculator = connectionCalculator;
	this.connections = new UniqueList<>();
    }

    public int getNeuronCount() {
	return neuronCount;
    }

    public void setNeuronCount(int neuronCount) {
	this.neuronCount = neuronCount;
    }

    public ConnectionCalculator getConnectionCalculator() {
	return connectionCalculator;
    }

    public void setConnectionCalculator(ConnectionCalculator connectionCalculator) {
	this.connectionCalculator = connectionCalculator;
    }

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
