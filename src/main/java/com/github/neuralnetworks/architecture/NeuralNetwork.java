package com.github.neuralnetworks.architecture;

import java.util.ArrayList;
import java.util.List;


/**
 * this is the base class for all the neural networks
 * @author hok
 *
 */
public class NeuralNetwork implements InputOutputLayers {

	protected Layer inputLayer;
	protected Layer outputLayer;
	protected List<Connections> connections;
	protected List<Layer> layers;

	public NeuralNetwork() {
		super();
		this.connections = new ArrayList<Connections>();
		this.layers = new ArrayList<Layer>();
	}

	public NeuralNetwork(Layer inputLayer, Layer outputLayer) {
		super();
		this.inputLayer = inputLayer;
		this.outputLayer = outputLayer;
		this.connections = new ArrayList<Connections>();
		this.layers = new ArrayList<Layer>();
	}

	@Override
	public Layer getInputLayer() {
		return inputLayer;
	}

	@Override
	public Layer getOutputLayer() {
		return outputLayer;
	}

	public List<Connections> getConnections() {
		return connections;
	}

	public void setConnections(List<Connections> connections) {
		this.connections = connections;
	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void setLayers(List<Layer> layers) {
		this.layers = layers;
	}
}
