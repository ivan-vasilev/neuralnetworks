package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Arrays;

import com.github.neuralnetworks.activation.ActivationFunction;
import com.github.neuralnetworks.neuroninput.InputFunction;

/**
 *
 * represents a layer of neurons
 *
 */
public class Layer implements Serializable {

	private static final long serialVersionUID = 1035633207383317489L;

	private int neuronCount;
	private InputFunction inputFunction;
	private ActivationFunction activationFunction;
	private Connections[] connectionGraphs;

	public Layer(int neuronCount) {
		super();
		this.neuronCount = neuronCount;
	}

	public Layer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction) {
		super();
		this.neuronCount = neuronCount;
		this.inputFunction = inputFunction;
		this.activationFunction = activationFunction;
	}

	public Layer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction, Connections[] connectionGraphs) {
		super();
		this.neuronCount = neuronCount;
		this.inputFunction = inputFunction;
		this.activationFunction = activationFunction;
		this.connectionGraphs = connectionGraphs;
	}

	public int getNeuronCount() {
		return neuronCount;
	}

	public void setNeuronCount(int neuronCount) {
		this.neuronCount = neuronCount;
	}

	public InputFunction getInputFunction() {
		return inputFunction;
	}

	public void setInputFunction(InputFunction inputFunction) {
		this.inputFunction = inputFunction;
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
