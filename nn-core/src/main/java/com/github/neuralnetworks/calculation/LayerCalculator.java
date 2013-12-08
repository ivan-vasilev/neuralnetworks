package com.github.neuralnetworks.calculation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * The implementations of this interface provide a way of propagating results
 * from one layer to the next
 */
public interface LayerCalculator {

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
     *            - where the results are written
     * @param layer
     *            - current layer
     */
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer);
}
