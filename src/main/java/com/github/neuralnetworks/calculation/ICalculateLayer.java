package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.IConnections;

/**
 *
 * the implementations of this interface provide a way of propagating results from one layer to the next
 *
 */
public interface ICalculateLayer {

	/**
	 * calculates
	 * @param precedingResult - previous layer result
	 * @param connections - connections between the previous layer and the next
	 * @return
	 */
	public CalculationResult calculate(CalculationResult precedingResult, IConnections connections);
}
