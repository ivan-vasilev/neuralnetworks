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
	private int neuronWeightsCount;

	public ConnectionGraph() {
		super();
	}

	public ConnectionGraph(float[] weights, int neuronWeightsCount) {
		super();
		this.weights = weights;
		this.neuronWeightsCount = neuronWeightsCount;
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
	 * @return count of connections for each neuron
	 */
	public int getNeuronWeightsCount() {
		return neuronWeightsCount;
	}

	public void setNeuronWeightsCount(int neuronWeightsCount) {
		this.neuronWeightsCount = neuronWeightsCount;
	}
}
