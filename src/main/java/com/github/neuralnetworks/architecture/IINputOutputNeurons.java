package com.github.neuralnetworks.architecture;


/**
 * this interface is implemented by everything that wants to present itself as a black box with with a list of input/output neurons
 * for example these could be whole neural network taking part in committee of machines or single convolutional layers
 * @author hok
 *
 */
public interface IINputOutputNeurons {
	public Neuron[] getInputNeurons();
	public Neuron[] getOutputNeurons();
}
