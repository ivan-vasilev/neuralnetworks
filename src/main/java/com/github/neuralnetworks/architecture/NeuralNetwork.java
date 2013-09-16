package com.github.neuralnetworks.architecture;


/**
 * this is the base class for all the neural networks
 * @author hok
 *
 */
public class NeuralNetwork implements IINputOutputLayers {

	protected Layer inputLayer;
	protected Layer outputLayer;

	public NeuralNetwork() {
		super();
	}

	public NeuralNetwork(Layer inputLayer, Layer outputLayer) {
		super();
		this.inputLayer = inputLayer;
		this.outputLayer = outputLayer;
	}

	@Override
	public Layer getInputLayer() {
		return inputLayer;
	}

	@Override
	public Layer getOutputLayer() {
		return outputLayer;
	}
}
