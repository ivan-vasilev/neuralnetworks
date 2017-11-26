package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Base interface for layer calculator. The difference with the feedforward layer calculator is the "activations" parameter, which contains the activations from the feedforward phse
 */
public interface BackPropagationLayerCalculator
{
	public void backpropagate(NeuralNetwork nn, ValuesProvider activations, ValuesProvider results);
}
