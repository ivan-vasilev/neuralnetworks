package com.github.neuralnetworks.architecture.types;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.activation.ActivationFunction;
import com.github.neuralnetworks.activation.RepeaterFunction;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.neuroninput.AparapiWeightedSum;
import com.github.neuralnetworks.neuroninput.ConstantInput;
import com.github.neuralnetworks.neuroninput.InputFunction;
import com.github.neuralnetworks.util.Constants;

/**
 * a Multi Layer perceptron network
 *
 * @author hok
 *
 */
public class MultiLayerPerceptron extends NeuralNetwork {

	/**
	 * @param layers - number of layers and number of neurons in each layer
	 */
	public MultiLayerPerceptron(List<Integer> layerProperties, Map<String, Object> properties) {
		super();
		if (layerProperties.size() < 2) {
			throw new IllegalArgumentException("The newtork must have at least one layer");
		}

		InputFunction inputFunction = properties.containsKey(Constants.INPUT_FUNCTION) ? (InputFunction) properties.get(Constants.INPUT_FUNCTION) : null;
		ActivationFunction activationFunction = properties.containsKey(Constants.ACTIVATION_FUNCTION) ? (ActivationFunction) properties.get(Constants.ACTIVATION_FUNCTION) : null;
		Boolean addBias = properties.containsKey(Constants.ADD_BIAS) ? (Boolean) properties.get(Constants.ADD_BIAS) : false;

		// populate input layer
		inputLayer = new Layer(layerProperties.get(0), inputFunction instanceof AparapiWeightedSum ? inputFunction : new AparapiWeightedSum(), new RepeaterFunction());
		layers.add(inputLayer);

		// layers are created
		for (int i = 1; i < layerProperties.size(); i++) {
			layers.add(new Layer(layerProperties.get(i), inputFunction, activationFunction));
			if (addBias) {
				Layer bias = new Layer(layerProperties.get(i), new ConstantInput(1), new RepeaterFunction());
				layers.add(bias);
				connections.add(new OneToOne(inputLayer, bias));
			}
		}

		outputLayer = layers.get(layers.size() - 1);

		// layer connections
		for (int i = 1; i < layers.size(); i++) {
			connections.add(new FullyConnected(layers.get(i - 1), layers.get(i)));
		}
	}
}
