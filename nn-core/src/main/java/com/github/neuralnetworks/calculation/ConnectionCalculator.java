package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations of this interface calculate a single connection between layers
 * !!! Important !!! The results of the calculations are represented as matrices (Matrix).
 * This is done, because it is assumed that implementations will provide a way for calculating many input results at once.
 * Each column of the matrix represents a single input. For example if the network is trained to classify MNIST images, each column of the input matrix will represent single MNIST image.
 */
public interface ConnectionCalculator extends Serializable {
    /**
     * @param connections - a map that contains the "input" connections to this connection. Each input connection comes with a Matrix that represents the input that comes from the layer on the other side of the link.
     * @param output - output matrix where the result is stored
     * @param targetLayer - the target layer, to which "output" is associated
     */
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer);
}
