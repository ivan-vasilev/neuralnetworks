package com.github.neuralnetworks.architecture;

/**
 *
 * represents a fully connected weight matrix between two layers of neurons
 *
 */
public class FullyConnected extends ConnectionsImpl {

	private ConnectionGraph connectionGraph;
	private int inputLayerStartNeuron;
	private int outputLayerStartNeuron;

	public FullyConnected(Layer inputLayer, Layer outputLayer) {
		super(inputLayer, outputLayer);

		// initialize input/output bindings
		outputLayerStartNeuron = inputLayerStartNeuron = 0;

		connectionGraph = new ConnectionGraph(new float[inputLayer.getNeuronCount() * outputLayer.getNeuronCount()], inputLayer.getNeuronCount());
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
	public ConnectionGraph getConnectionGraph() {
		return connectionGraph;
	}
}
