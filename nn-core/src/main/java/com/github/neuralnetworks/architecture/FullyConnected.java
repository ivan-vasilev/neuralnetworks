package com.github.neuralnetworks.architecture;

/**
 * Fully connected weight matrix between two layers of neurons
 */
public class FullyConnected extends ConnectionsImpl implements GraphConnections {

    private static final long serialVersionUID = 1L;

    /**
     * Weight matrix for the weights of the links
     */
    private final Matrix connectionGraph;

    public FullyConnected(Layer inputLayer, Layer outputLayer, int inputUnitCount, int outputUnitCount) {
	super(inputLayer, outputLayer);

	// connection graph is initialized depending on the size of the input/output layers
	connectionGraph = new Matrix(new float[inputUnitCount * outputUnitCount], inputUnitCount);
    }

    @Override
    public Matrix getConnectionGraph() {
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
