package com.github.neuralnetworks.architecture;

/**
 * Fully connected weight matrix between two layers of neurons
 */
public class FullyConnected extends ConnectionsImpl implements GraphConnections {

    /**
     * Weight matrix for the weights of the links
     */
    private final Matrix connectionGraph;

    public FullyConnected(Layer inputLayer, Layer outputLayer) {
	super(inputLayer, outputLayer);

	// connection graph is initialized depending on the size of the input/output layers
	connectionGraph = new Matrix(new float[inputLayer.getNeuronCount() * outputLayer.getNeuronCount()], inputLayer.getNeuronCount());
    }

    @Override
    public Matrix getConnectionGraph() {
	return connectionGraph;
    }
}
