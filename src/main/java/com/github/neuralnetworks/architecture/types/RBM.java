package com.github.neuralnetworks.architecture.types;

import java.util.Map;

import com.github.neuralnetworks.activation.ActivationFunction;
import com.github.neuralnetworks.activation.RepeaterFunction;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.neuroninput.ConstantInput;
import com.github.neuralnetworks.neuroninput.InputFunction;
import com.github.neuralnetworks.util.Constants;

/**
 * Restricted Boltzmann Machine
 *
 */
public class RBM extends NeuralNetwork {

	/**
	 * @param layers - number of layers and number of neurons in each layer
	 */
	public RBM(Map<String, Object> properties) {
		super();

		int visibleCount = (Integer) properties.get(Constants.VISIBLE_COUNT);
		int hiddenCount = (Integer) properties.get(Constants.HIDDEN_COUNT);
		InputFunction inputFunction = (InputFunction) properties.get(Constants.INPUT_FUNCTION);
		ActivationFunction activationFunction = (ActivationFunction) properties.get(Constants.ACTIVATION_FUNCTION);
		Boolean addBias = (Boolean) properties.get(Constants.ADD_BIAS);

		// populate visible layer
		inputLayer = new Layer(visibleCount, inputFunction, activationFunction);
		layers.add(inputLayer);

		outputLayer = new Layer(hiddenCount, inputFunction, activationFunction);
		layers.add(outputLayer);

		if (addBias) {
			Layer visibleBias = new Layer(visibleCount, new ConstantInput(1), new RepeaterFunction());
			layers.add(visibleBias);
			connections.add(new OneToOne(inputLayer, visibleBias));

			Layer hiddenBias = new Layer(hiddenCount, new ConstantInput(1), new RepeaterFunction());
			layers.add(hiddenBias);
			connections.add(new OneToOne(hiddenBias, inputLayer));
		}

		// layer connections
		connections.add(new FullyConnected(inputLayer, outputLayer));
	}
}
