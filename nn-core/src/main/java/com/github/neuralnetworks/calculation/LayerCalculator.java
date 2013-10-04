package com.github.neuralnetworks.calculation;

import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;

/**
 *
 * the implementations of this interface provide a way of propagating results from one layer to the next
 *
 */
public interface LayerCalculator {

	/**
	 * @param calculatedLayers - existing results
	 * @param layer - the layer to be calculated
	 */
	public void calculate(Map<Layer, float[]> calculatedLayers, Layer layer);
}
