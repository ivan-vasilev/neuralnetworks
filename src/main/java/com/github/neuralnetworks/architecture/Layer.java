package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.architecture.activation.ActivationFunction;
import com.github.neuralnetworks.architecture.input.InputFunction;

/**
 *
 * represents a layer of neurons
 *
 */
public class Layer {
	private int neuronCount;
	private InputFunction inputFunction;
	private ActivationFunction activationFunction;
	private IConnections[] inboundConnectionGraphs;
	private IConnections[] outboundConnectionGraphs;

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

}
