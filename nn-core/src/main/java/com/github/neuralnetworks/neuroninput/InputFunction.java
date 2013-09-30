package com.github.neuralnetworks.neuroninput;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Connections;

/**
 * this interface is implemented by neuron input functions
 *
 * @author hok
 *
 */
public interface InputFunction extends Serializable {
	public void calculateForward(Connections graph, float[] inputValues, float[] result);
	public void calculateBackward(Connections graph, float[] inputValues, float[] result);
}
