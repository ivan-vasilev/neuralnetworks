package com.github.neuralnetworks.architecture;

/**
 *
 * represents a single dimension array for one-to-one link between layers (for example for use with biases)
 *
 */
public class OneToOne extends ConnectionsImpl {

	private ConnectionGraph connectionGraph;
	private int inputLayerStartNeuron;
	private int outputLayerStartNeuron;

	public OneToOne(Layer inputLayer, Layer outputLayer) {
		super(inputLayer, outputLayer);

		// initialize input/output bindings
		outputLayerStartNeuron = inputLayerStartNeuron = 0;

		if (inputLayer.getNeuronCount() != outputLayer.getNeuronCount()) {
			throw new IllegalArgumentException("layers must have equal neurons count");
		}

		// initialize weights array
		connectionGraph = new ConnectionGraph(new float[inputLayer.getNeuronCount()], 1);
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
