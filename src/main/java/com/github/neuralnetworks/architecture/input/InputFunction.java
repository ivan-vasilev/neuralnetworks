package com.github.neuralnetworks.architecture.input;

import java.io.Serializable;

/**
 * this interface is implemented by neuron input functions
 *
 * @author hok
 *
 */
public interface InputFunction extends Serializable {
	public double calc(double[] values, double[] weights);
}
