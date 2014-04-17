package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Fully connected weight matrix between two layers of neurons
 */
public class FullyConnected extends ConnectionsImpl implements WeightsConnections {

    private static final long serialVersionUID = 1L;

    /**
     * Weight matrix for the weights of the links
     */
    private final Matrix connectionGraph;

    public FullyConnected(Layer inputLayer, Layer outputLayer, int inputUnitCount, int outputUnitCount) {
	this(inputLayer, outputLayer, TensorFactory.matrix(new float[inputUnitCount * outputUnitCount], inputUnitCount));
    }

    public FullyConnected(Layer inputLayer, Layer outputLayer, Matrix connectionGraph) {
	super(inputLayer, outputLayer);
	this.connectionGraph = connectionGraph;
    }

    @Override
    public Matrix getWeights() {
	return connectionGraph;
    }

    @Override
    public int getInputUnitCount() {
	return connectionGraph.getColumns();
    }

    @Override
    public int getOutputUnitCount() {
	return connectionGraph.getRows();
    }
}
