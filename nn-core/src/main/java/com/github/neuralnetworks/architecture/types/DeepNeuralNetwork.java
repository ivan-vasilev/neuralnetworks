package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.NeuralNetwork;

/**
 * Base interface for all deep neural networks.
 * Each deep network provides a list of neural networks. For example Stacked autoencoder would return a list of Autoencoders.
 */
public interface DeepNeuralNetwork<N extends NeuralNetwork> extends NeuralNetwork {
    public List<N> getNeuralNetworks();
}
