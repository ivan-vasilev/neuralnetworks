package com.github.neuralnetworks.architecture;


public class NeuronConnections {

	private Neuron neuron;
	private double[] weights;
	private Neuron[] neurons;

	public NeuronConnections(Neuron neuron, double[] weights, Neuron[] neurons) {
		super();
		this.neuron = neuron;
		this.weights = weights;
		this.neurons = neurons;
	}

	public Neuron getNeuron() {
		return neuron;
	}

	public double[] getWeights() {
		return weights;
	}

	public Neuron[] getNeurons() {
		return neurons;
	}
}
