package com.github.neuralnetworks.architecture;


/**
 * this is the base class for all the neural networks
 * @author hok
 *
 */
public class NeuralNetwork implements IINputOutputNeurons {

	protected Neuron[] inputNeurons;
	protected Neuron[] outputNeurons;

	public NeuralNetwork() {
		super();
	}

	public NeuralNetwork(Neuron[] inputNeurons, Neuron[] outputNeurons) {
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
