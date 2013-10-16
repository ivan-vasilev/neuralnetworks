package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.neuronfunctions.ActivationFunction;
import com.github.neuralnetworks.neuronfunctions.InputFunction;

/**
 * 
 * represents a layer of neurons
 * 
 */
public class Layer implements Serializable {

    private static final long serialVersionUID = 1035633207383317489L;

    private int neuronCount;
    private InputFunction forwardInputFunction;
    private InputFunction backwardInputFunction;
    private ActivationFunction activationFunction;
    private Set<Connections> connections;

    public Layer(int neuronCount) {
	super();
	this.neuronCount = neuronCount;
	this.connections = new HashSet<>();
    }

    public Layer(int neuronCount, InputFunction forwardInputFunction, InputFunction backwardInputFunction, ActivationFunction activationFunction) {
	super();
	this.neuronCount = neuronCount;
	this.forwardInputFunction = forwardInputFunction;
	this.backwardInputFunction = backwardInputFunction;
	this.activationFunction = activationFunction;
	this.connections = new HashSet<>();
    }

    public Layer(int neuronCount, InputFunction forwardInputFunction, InputFunction backwardInputFunction, ActivationFunction activationFunction, Set<Connections> connections) {
	super();
	this.neuronCount = neuronCount;
	this.forwardInputFunction = forwardInputFunction;
	this.backwardInputFunction = backwardInputFunction;
	this.activationFunction = activationFunction;
	this.connections = connections;
    }

    public int getNeuronCount() {
	return neuronCount;
    }

    public void setNeuronCount(int neuronCount) {
	this.neuronCount = neuronCount;
    }

    public InputFunction getForwardInputFunction() {
	return forwardInputFunction;
    }

    public void setForwardInputFunction(InputFunction forwardInputFunction) {
	this.forwardInputFunction = forwardInputFunction;
    }

    public InputFunction getBackwardInputFunction() {
	return backwardInputFunction;
    }

    public void setBackwardInputFunction(InputFunction backwardInputFunction) {
	this.backwardInputFunction = backwardInputFunction;
    }

    public ActivationFunction getActivationFunction() {
	return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
	this.activationFunction = activationFunction;
    }

    public Set<Connections> getConnections() {
	return connections;
    }

    public void setConnections(Set<Connections> connections) {
	this.connections = connections;
    }

    public void addConnection(Connections connection) {
	if (connections == null) {
	    connections = new HashSet<>();
	}

	connections.add(connection);
    }
}
