package com.github.neuralnetworks.architecture;

/**
 * Fully connected weight matrix between two layers of neurons
 */
public class FullyConnected extends ConnectionsImpl implements GraphConnections {

    /**
     * Weight matrix for the weights of the links
     */
    private final Matrix connectionGraph;

    /**
     * This property allows the weights to start from given neuron in the input layer. This allows for combining multiple outputs for one layer
     */
    private int inputLayerStartNeuron;

    /**
     * This property allows the weights to start from given neuron in the output layer. This allows for combining multiple outputs for one layer
     */
    private final int outputLayerStartNeuron;

    public FullyConnected(Layer inputLayer, Layer outputLayer) {
	super(inputLayer, outputLayer);

	// initialize input/output bindings
	outputLayerStartNeuron = inputLayerStartNeuron = 0;

	// connection graph is initialized depending on the size of the input/output layers
	connectionGraph = new Matrix(new float[inputLayer.getNeuronCount() * outputLayer.getNeuronCount()], inputLayer.getNeuronCount());
    }

    @Override
    public int getInputLayerStartNeuron() {
	return inputLayerStartNeuron;
    }

    @Override
    public int getOutputLayerStartNeuron() {
	return outputLayerStartNeuron;
    }

    @Override
    public Matrix getConnectionGraph() {
	return connectionGraph;
    }
}
