package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.Arrays;

import com.github.neuralnetworks.architecture.activation.ActivationFunction;
import com.github.neuralnetworks.architecture.input.InputFunction;

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
	private IConnections[] inboundConnectionGraphs;
	private IConnections[] outboundConnectionGraphs;

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

	public Layer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction, IConnections[] inboundConnectionGraphs, IConnections[] outboundConnectionGraphs) {
		super();
		this.neuronCount = neuronCount;
		this.inputFunction = inputFunction;
		this.activationFunction = activationFunction;
		this.inboundConnectionGraphs = inboundConnectionGraphs;
		this.outboundConnectionGraphs = outboundConnectionGraphs;
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

	public IConnections[] getInboundConnectionGraphs() {
		return inboundConnectionGraphs;
	}

	public void setInboundConnectionGraphs(IConnections[] inboundConnectionGraphs) {
		this.inboundConnectionGraphs = inboundConnectionGraphs;
	}

	public IConnections[] getOutboundConnectionGraphs() {
		return outboundConnectionGraphs;
	}

	public void setOutboundConnectionGraphs(IConnections[] outboundConnectionGraphs) {
		this.outboundConnectionGraphs = outboundConnectionGraphs;
	}

	public void addInboundConnectionGraph(IConnections graph) {
		if (inboundConnectionGraphs == null) {
			inboundConnectionGraphs = new IConnections[1];
		} else {
			inboundConnectionGraphs = Arrays.copyOf(inboundConnectionGraphs, inboundConnectionGraphs.length + 1);
		}

		inboundConnectionGraphs[inboundConnectionGraphs.length - 1] = graph;
	}

	public void addOutboundConnectionGraph(IConnections graph) {
		if (outboundConnectionGraphs == null) {
			outboundConnectionGraphs = new IConnections[1];
		} else {
			outboundConnectionGraphs = Arrays.copyOf(outboundConnectionGraphs, outboundConnectionGraphs.length + 1);
		}

		outboundConnectionGraphs[outboundConnectionGraphs.length - 1] = graph;
	}
}
