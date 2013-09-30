package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
	private Connections[] inboundConnectionGraphs;
	private Connections[] outboundConnectionGraphs;

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

	public Layer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction, Connections[] inboundConnectionGraphs, Connections[] outboundConnectionGraphs) {
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

	public Connections[] getInboundConnectionGraphs() {
		return inboundConnectionGraphs;
	}

	public void setInboundConnectionGraphs(Connections[] inboundConnectionGraphs) {
		this.inboundConnectionGraphs = inboundConnectionGraphs;
	}

	public Connections[] getOutboundConnectionGraphs() {
		return outboundConnectionGraphs;
	}

	public void setOutboundConnectionGraphs(Connections[] outboundConnectionGraphs) {
		this.outboundConnectionGraphs = outboundConnectionGraphs;
	}

	public void addInboundConnectionGraph(Connections graph) {
		if (inboundConnectionGraphs == null) {
			inboundConnectionGraphs = new Connections[1];
		} else {
			inboundConnectionGraphs = Arrays.copyOf(inboundConnectionGraphs, inboundConnectionGraphs.length + 1);
		}

		inboundConnectionGraphs[inboundConnectionGraphs.length - 1] = graph;
	}

	public void addOutboundConnectionGraph(Connections graph) {
		if (outboundConnectionGraphs == null) {
			outboundConnectionGraphs = new Connections[1];
		} else {
			outboundConnectionGraphs = Arrays.copyOf(outboundConnectionGraphs, outboundConnectionGraphs.length + 1);
		}

		outboundConnectionGraphs[outboundConnectionGraphs.length - 1] = graph;
	}

	public List<Layer> getAdjacentOutputLayers() {
		List<Layer> result = new ArrayList<Layer>();
		if (getOutboundConnectionGraphs() != null) {
			for (Connections c : getOutboundConnectionGraphs()) {
				result.add(c.getOutputLayer());
			}
		}

		return result;
	}

	public List<Layer> getAdjacentInputLayers() {
		List<Layer> result = new ArrayList<Layer>();
		if (getInboundConnectionGraphs() != null) {
			for (Connections c : getInboundConnectionGraphs()) {
				result.add(c.getOutputLayer());
			}
		}

		return result;
	}
}
