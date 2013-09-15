package com.github.neuralnetworks.architecture;


public class NeuronConnections {

	private Neuron neuron;
	private float[] weights;
	private Neuron[] neurons;

	public NeuronConnections(Neuron neuron, float[] weights, Neuron[] neurons) {
		super();
		this.neuron = neuron;
		this.weights = weights;
		this.neurons = neurons;
	}

	public Neuron getNeuron() {
		return neuron;
	}

	public float[] getWeights() {
		return weights;
	}

	public Neuron[] getNeurons() {
		return neurons;
	}
}
