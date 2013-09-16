package com.github.neuralnetworks.architecture;


/**
 * this abstract class serves as a base for all weight matrices
 *
 * @author hok
 *
 */
public abstract class Connections implements IConnections {

	/**
	 * input layer of neurons
	 */
	protected Layer inputLayer;

	/**
	 * output layer
	 */
	protected Layer outputLayer;

	public Connections(Layer inputLayer, Layer outputLayer) {
		super();
		this.inputLayer = inputLayer;
		this.outputLayer = outputLayer;

		inputLayer.addOutboundConnectionGraph(this);
		outputLayer.addInboundConnectionGraph(this);
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
