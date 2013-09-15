package com.github.neuralnetworks.architecture;


/**
 * represents a fully connected weight matrix between two layers of neurons
 *
 * @author hok
 *
 */
public class FullyConnected extends Connections {

	protected float[][] weightMatrix;

	public FullyConnected(Neuron[] inputNeurons, Neuron[] outputNeurons) {
		super(inputNeurons, outputNeurons);
		weightMatrix = new float[outputNeurons.length][inputNeurons.length];
	}

	@Override
	public NeuronConnections getConnections(Neuron n) {
		NeuronConnections result = null;
		int index = n.getLayerIndex();
		if (index < outputNeurons.length && outputNeurons[index] == n) {
			result = new NeuronConnections(n, weightMatrix[index], inputNeurons);
		} else if (index < inputNeurons.length && inputNeurons[index] == n) {
			float[] weights = new float[outputNeurons.length];
			for (int i = 0; i < outputNeurons.length; i++) {
				weights[i] = weightMatrix[index][i];
			}

			result = new NeuronConnections(n, weights, outputNeurons);
		}

		return result;
	}

	public float[][] getWeightMatrix() {
		return weightMatrix;
	}

	public void setWeightMatrix(float[][] weightMatrix) {
		this.weightMatrix = weightMatrix;
	}
}
