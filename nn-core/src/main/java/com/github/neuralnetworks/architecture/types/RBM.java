package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.activation.ActivationFunction;
import com.github.neuralnetworks.activation.RepeaterFunction;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.neuroninput.ConstantInput;
import com.github.neuralnetworks.neuroninput.InputFunction;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 *
 * Restricted Boltzmann Machine
 *
 */
public class RBM extends NeuralNetwork {

	private Connections mainConnections;
	private OneToOne visibleBiasConnections;
	private OneToOne hiddenBiasConnections;
	private Layer hiddenLayer;

	public RBM(Properties properties) {
		super(properties);

		int visibleCount = properties.getParameter(Constants.VISIBLE_COUNT);
		int hiddenCount = properties.getParameter(Constants.HIDDEN_COUNT);
		InputFunction inputFunction = properties.getParameter(Constants.INPUT_FUNCTION);
		ActivationFunction activationFunction = properties.getParameter(Constants.ACTIVATION_FUNCTION);
		boolean addBias = properties.containsKey(Constants.ADD_BIAS) ? (boolean) properties.getParameter(Constants.ADD_BIAS) : false;

		// populate visible layer
		outputLayer = inputLayer = new Layer(visibleCount, inputFunction, activationFunction);
		layers.add(inputLayer);

		hiddenLayer = new Layer(hiddenCount, inputFunction, activationFunction);
		layers.add(hiddenLayer);

		if (addBias) {
			Layer visibleBiasLayer = new Layer(visibleCount, new ConstantInput(1), new RepeaterFunction());
			layers.add(visibleBiasLayer);
			connections.add(visibleBiasConnections = new OneToOne(inputLayer, visibleBiasLayer));

			Layer hiddenBiasLayer = new Layer(hiddenCount, new ConstantInput(1), new RepeaterFunction());
			layers.add(hiddenBiasLayer);
			connections.add(hiddenBiasConnections = new OneToOne(hiddenBiasLayer, inputLayer));
		}

		// layer connections
		connections.add(mainConnections = new FullyConnected(inputLayer, outputLayer));
	}

	public Connections getMainConnections() {
		return mainConnections;
	}

	public Connections getVisibleBiasConnections() {
		return visibleBiasConnections;
	}

	public Connections getHiddenBiasConnections() {
		return hiddenBiasConnections;
	}

	public Layer getVisibleLayer() {
		return this.inputLayer;
	}

	public Layer getHiddenLayer() {
		return this.hiddenLayer;
	}
}
