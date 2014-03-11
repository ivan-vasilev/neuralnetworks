package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;

/**
 * The implementations of this interface provide a way of propagating results
 * from one layer to the next
 */
public interface LayerCalculator extends Serializable {

    /**
     * This method receives a target layer to calculate, a list of calculated layer results and a list of layers that have already been calculated.
     * This information is enough for the target layer can be calculated
     * 
     * @param calculatedLayers
     *            - existing results
     * @param layer
     *            - the layer to be calculated
     */
    /**
     * @param calculatedLayers
     *            - calculated layers that are provided as input
     * @param results
     *            - where the results are stored
     * @param layer
     *            - current layer
     * @param neuralNetwork
     *            - the network of context for calculation
     */
    public void calculate(NeuralNetwork neuralNetwork, Layer layer, Set<Layer> calculatedLayers, ValuesProvider results);
}
