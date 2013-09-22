package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Layer;

/**
 *
 * implementations of this interface act as a factory for layer calculators
 *
 */
public interface LayerCalculatorProvider {
	public LayerCalculator getCalculator(Layer layer);
}
