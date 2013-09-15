package com.github.neuralnetworks.calculation;

import java.util.Map;

import com.github.neuralnetworks.architecture.Neuron;

/**
 *
 * the implementations of this interface provide a way of propagating results from one layer to the next
 *
 */
public interface ICalculateLayer {

	/**
	 * @param calculatedLayers - existing results
	 * @param layer - the layer to be calculated
	 */
	public float[] calculateForward(Map<Neuron[], float[]> calculatedLayers, Neuron[] layer);

	/**
	 * @param calculatedLayers - existing layers
	 * @param layer - the layer to be calculated
	 */
	public float[] calculateBackward(Map<Neuron[], float[]> calculatedLayers, Neuron[] layer);
}
