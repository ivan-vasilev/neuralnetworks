package com.github.neuralnetworks.architecture;


/**
 * this interface is implemented by everything that wants to present itself as a black box with with a list of input/output layers
 * for example these could be whole neural network taking part in committee of machines or single convolutional layers
 * @author hok
 *
 */
public interface InputOutputLayers {
	public Layer getInputLayer();
	public Layer getOutputLayer();
}
