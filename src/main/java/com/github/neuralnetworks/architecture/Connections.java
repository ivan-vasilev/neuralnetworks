package com.github.neuralnetworks.architecture;


/**
 * this abstract class serves as a base for all weight matrices
 *
 * @author hok
 *
 */
public abstract class Connections implements IConnections {

	/**
	 * represents a list of input (in terms of the neural network architecture) neurons
	 */
	protected Neuron[] inputNeurons;

	/**
	 * represents a list of output (in terms of the neural network architecture) neurons
	 */
	protected Neuron[] outputNeurons;

	public Connections(Neuron[] inputNeurons, Neuron[] outputNeurons) {
		super();
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
	}

	@Override
	public Neuron[] getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neuron[] getOutputNeurons() {
		return outputNeurons;
	}
}
