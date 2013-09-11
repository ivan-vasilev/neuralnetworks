package com.github.neuralnetworks.architecture;


/**
 * represents a fully connected weight matrix between two layers of neurons
 *
 * @author hok
 *
 */
public class FullyConnected extends Connections {

	protected double[][] weightMatrix;

	public FullyConnected(Neuron[] inputNeurons, Neuron[] outputNeurons) {
		super(inputNeurons, outputNeurons);
		weightMatrix = new double[outputNeurons.length][inputNeurons.length];
	}

	@Override
	public NeuronConnections getConnections(Neuron n) {
		NeuronConnections result = null;
		int index = n.getLayerIndex();
		if (index < outputNeurons.length && outputNeurons[index] == n) {
			result = new NeuronConnections(n, weightMatrix[index], inputNeurons);
		} else if (index < inputNeurons.length && inputNeurons[index] == n) {
			double[] weights = new double[outputNeurons.length];
			for (int i = 0; i < outputNeurons.length; i++) {
				weights[i] = weightMatrix[index][i];
			}

			result = new NeuronConnections(n, weights, outputNeurons);
		}

		return result;
	}

	public double[][] getWeightMatrix() {
		return weightMatrix;
	}

	public void setWeightMatrix(double[][] weightMatrix) {
		this.weightMatrix = weightMatrix;
	}
}
