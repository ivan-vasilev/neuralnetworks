package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Arrays;

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
	private Connections[] connectionGraphs;

	public Layer(int neuronCount) {
		super();
		this.neuronCount = neuronCount;
	}

	public Layer(int neuronCount, InputFunction forwardInputFunction, InputFunction backwardInputFunction, ActivationFunction activationFunction) {
		super();
		this.neuronCount = neuronCount;
		this.forwardInputFunction = forwardInputFunction;
		this.backwardInputFunction = backwardInputFunction;
		this.activationFunction = activationFunction;
	}

	public Layer(int neuronCount, InputFunction forwardInputFunction, InputFunction backwardInputFunction, ActivationFunction activationFunction, Connections[] connectionGraphs) {
		super();
		this.neuronCount = neuronCount;
		this.forwardInputFunction = forwardInputFunction;
		this.backwardInputFunction = backwardInputFunction;
		this.activationFunction = activationFunction;
		this.connectionGraphs = connectionGraphs;
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

	public Connections[] getConnectionGraphs() {
		return connectionGraphs;
	}

	public void setConnectionGraphs(Connections[] connectionGraphs) {
		this.connectionGraphs = connectionGraphs;
	}

	public void addConnectionGraph(Connections graph) {
		if (connectionGraphs == null) {
			connectionGraphs = new Connections[1];
		} else {
			connectionGraphs = Arrays.copyOf(connectionGraphs, connectionGraphs.length + 1);
		}

		connectionGraphs[connectionGraphs.length - 1] = graph;
	}
}
