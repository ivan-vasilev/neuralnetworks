package com.github.neuralnetworks.architecture;


/**
 *
 * represents a fully connected weight matrix between two layers of neurons
 *
 */
public class FullyConnected extends Connections {

	private ConnectionGraph forwardConnections;
	private ConnectionGraph backwardConnections;
	private int[] inputLayerNeurons;
	private int[] outputLayerNeurons;

	public FullyConnected(Layer inputLayer, Layer outputLayer) {
		super(inputLayer, outputLayer);

		// initialize input/output bindings
		inputLayerNeurons = new int[inputLayer.getNeuronCount()];
		for (int i = 0; i < inputLayerNeurons.length; i++) {
			inputLayerNeurons[i] = i;
		}

		outputLayerNeurons = new int[outputLayer.getNeuronCount()];
		for (int i = 0; i < outputLayerNeurons.length; i++) {
			outputLayerNeurons[i] = i;
		}

		// initialize weights array
		float[] weights = new float[inputLayer.getNeuronCount() * outputLayer.getNeuronCount()];

		// initialize forward propagation graph
		int[] neuronWeightsStartPosition = new int[inputLayer.getNeuronCount()];
		for (int i = 0; i < neuronWeightsStartPosition.length; i++) {
			neuronWeightsStartPosition[i] = outputLayer.getNeuronCount() * i;
		}

		int[] neuronWeightsCount = new int[inputLayer.getNeuronCount()];
		for (int i = 0; i < neuronWeightsCount.length; i++) {
			neuronWeightsCount[i] = outputLayer.getNeuronCount();
		}

		forwardConnections = new ConnectionGraph(weights, neuronWeightsStartPosition, neuronWeightsCount, 1);

		// initialize backward propagation graph
		neuronWeightsStartPosition = new int[outputLayer.getNeuronCount()];
		for (int i = 0; i < neuronWeightsStartPosition.length; i++) {
			neuronWeightsStartPosition[i] = outputLayer.getNeuronCount() * i;
		}

		neuronWeightsCount = new int[outputLayer.getNeuronCount()];
		for (int i = 0; i < neuronWeightsCount.length; i++) {
			neuronWeightsCount[i] = inputLayer.getNeuronCount();
		}

		backwardConnections = new ConnectionGraph(weights, neuronWeightsStartPosition, neuronWeightsCount, 1);
	}

	@Override
	public int[] getInputLayerNeurons() {
		return inputLayerNeurons;
	}

	@Override
	public int[] getOutputLayerNeurons() {
		return outputLayerNeurons;
	}

	@Override
	public ConnectionGraph getForwardConnectionGraph() {
		return forwardConnections;
	}

	@Override
	public ConnectionGraph getBackwardConnectionGraph() {
		return backwardConnections;
	}
}
