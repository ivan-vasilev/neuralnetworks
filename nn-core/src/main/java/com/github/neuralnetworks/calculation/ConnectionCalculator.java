package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;

/**
 * Implementations of this interface calculate a single connection between layers
 * !!! Important !!! The results of the calculations are represented as tensors (Tensor).
 * This is done, because it is assumed that implementations will provide a way for calculating many input results at once.
 * Each column of the matrix represents a single input. For example if the network is trained to classify MNIST images, each column of the input matrix will represent single MNIST image.
 */
public interface ConnectionCalculator extends Serializable {
    /**
     * @param connections - list of connections to calculate
     * @param valuesProvider - values provider for the connections
     * @param targetLayer - the target layer, to which "output" is associated
     */
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer);
}
