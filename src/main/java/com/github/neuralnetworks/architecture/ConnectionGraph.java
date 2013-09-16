package com.github.neuralnetworks.architecture;

import java.io.Serializable;

/**
 *
 * this is a graph representation based on a one dimensional array (in order to
 * facilitate gpu computations)
 *
 */
public class ConnectionGraph implements Serializable {

	private static final long serialVersionUID = 1L;

	private float[] weights;
	private int[] neuronWeightsStartPosition;
	private int[] neuronWeightsCount;
	private int neuronWeightsStep;

	public ConnectionGraph() {
		super();
	}

	public ConnectionGraph(float[] weights, int[] neuronWeightsStartPosition, int[] neuronWeightsCount, int neuronWeightsStep) {
		super();
		this.weights = weights;
		this.neuronWeightsStartPosition = neuronWeightsStartPosition;
		this.neuronWeightsCount = neuronWeightsCount;
		this.neuronWeightsStep = neuronWeightsStep;
	}

	/**
	 * @return list of weights in form of a single array (this is structure is
	 *         chosen to enable gpu processing)
	 */
	public float[] getWeights() {
		return weights;
	}

	public void setWeights(float[] weights) {
		this.weights = weights;
	}

	/**
	 * @return start position of connections for each neuron in the weights
	 *         array
	 */
	public int[] getNeuronWeightsStartPosition() {
		return neuronWeightsStartPosition;
	}

	public void setNeuronWeightsStartPosition(int[] neuronWeightsStartPosition) {
		this.neuronWeightsStartPosition = neuronWeightsStartPosition;
	}

	/**
	 * @return count of connections for each neuron
	 */
	public int[] getNeuronWeightsCount() {
		return neuronWeightsCount;
	}

	public void setNeuronWeightsCount(int[] neuronWeightsCount) {
		this.neuronWeightsCount = neuronWeightsCount;
	}

	/**
	 * @return the step in the weights array which separates single neuron's
	 *         weights
	 */
	public int getNeuronWeightsStep() {
		return neuronWeightsStep;
	}

	public void setNeuronWeightsStep(int neuronWeightsStep) {
		this.neuronWeightsStep = neuronWeightsStep;
	}
}
