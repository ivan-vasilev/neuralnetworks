package com.github.neuralnetworks.calculation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

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
	/**
	 * @param calculatedLayers - calculated layers that are provided as input
	 * @param results - where the results are written
	 * @param layer - current layer
	 */
	public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer);
}
