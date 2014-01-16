package com.github.neuralnetworks.architecture;

import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.calculation.LayerCalculator;

/**
 * this interface is implemented by everything that wants to present itself as a
 * black box with with a list of input/output layers for example these could be
 * whole neural network taking part in committee of machines, single
 * convolutional/subsamplingo layers or even a single connection between the layers
 */
public interface NeuralNetwork {

    /**
     * input layer
     */
    public Layer getInputLayer();

    /**
     * @return the layer that is used for output when the network is "stacked"
     */
    public Layer getOutputLayer();

    /**
     * @return the output layer for the rest of the world - this and the previous methods may return different results (in case of Autoencoders for example)
     */
    public Layer getDataOutputLayer();

    /**
     * @return all the layers in this network
     */
    public Set<Layer> getLayers();

    /**
     * @return all the connections in this network
     */
    public List<Connections> getConnections();

    /**
     * LayerCalculator associated to this network
     */
    public LayerCalculator getLayerCalculator();
}
