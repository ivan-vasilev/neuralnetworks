package com.github.neuralnetworks.architecture;

import java.util.Set;

/**
 * this interface is implemented by everything that wants to present itself as a
 * black box with with a list of input/output layers for example these could be
 * whole neural network taking part in committee of machines or single
 * convolutional layers
 */
public interface NeuralNetwork {
    public Layer getInputLayer();
    public Layer getOutputLayer();
    public Set<Layer> getLayers();
    public Set<Connections> getConnections();
}
