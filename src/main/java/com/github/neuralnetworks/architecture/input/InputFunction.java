package com.github.neuralnetworks.architecture.input;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.IConnections;

/**
 * this interface is implemented by neuron input functions
 *
 * @author hok
 *
 */
public interface InputFunction extends Serializable {
	public void calculateForward(IConnections graph, float[] inputValues, float[] result);
	public void calculateBackward(IConnections graph, float[] inputValues, float[] result);
}
