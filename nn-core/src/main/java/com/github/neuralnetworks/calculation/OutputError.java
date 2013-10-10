package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Matrix;

/**
 *
 * interface for calculating output error
 *
 */
public interface OutputError {
	public void addItem(Matrix networkOutput, Object[] targetOutput);
	public float getTotalNetworkError();
}
