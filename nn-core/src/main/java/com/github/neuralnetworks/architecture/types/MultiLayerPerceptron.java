package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.neuronfunctions.ActivationFunction;
import com.github.neuralnetworks.neuronfunctions.ConstantInput;
import com.github.neuralnetworks.neuronfunctions.InputFunction;
import com.github.neuralnetworks.neuronfunctions.RepeaterFunction;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * a Multi Layer perceptron network
 * 
 * @author hok
 * 
 */
public class MultiLayerPerceptron extends NeuralNetwork {

    /**
     * @param layers
     *            - number of layers and number of neurons in each layer
     */
    public MultiLayerPerceptron(Properties properties) {
	super(properties);

	@SuppressWarnings("unchecked")
	List<Integer> layerProperties = (List<Integer>) properties.get(Constants.LAYERS);
	if (layerProperties.size() < 2) {
	    throw new IllegalArgumentException("The newtork must have at least one layer");
	}

	InputFunction forwardInputFunction = (InputFunction) properties.get(Constants.FORWARD_INPUT_FUNCTION);
	InputFunction backwardInputFunction = (InputFunction) properties.get(Constants.BACKWARD_INPUT_FUNCTION);
	ActivationFunction activationFunction = (ActivationFunction) properties.get(Constants.ACTIVATION_FUNCTION);
	Boolean addBias = properties.containsKey(Constants.ADD_BIAS) ? (Boolean) properties.get(Constants.ADD_BIAS) : false;

	// populate input layer
	inputLayer = new Layer(layerProperties.get(0), forwardInputFunction, backwardInputFunction, activationFunction);
	layers.add(inputLayer);

	// layers are created
	for (int i = 1; i < layerProperties.size(); i++) {
	    layers.add(new Layer(layerProperties.get(i), forwardInputFunction, backwardInputFunction, activationFunction));
	    if (addBias) {
		Layer bias = new Layer(layerProperties.get(i), new ConstantInput(1), new ConstantInput(1), new RepeaterFunction());
		layers.add(bias);
		connections.add(new OneToOne(bias, inputLayer));
	    }
	}

	outputLayer = layers.get(layers.size() - 1);

	// layer connections
	for (int i = 1; i < layers.size(); i++) {
	    connections.add(new FullyConnected(layers.get(i - 1), layers.get(i)));
	}
    }
}
