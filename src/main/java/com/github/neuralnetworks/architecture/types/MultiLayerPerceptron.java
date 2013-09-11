package com.github.neuralnetworks.architecture.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.IConnections;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Neuron;
import com.github.neuralnetworks.architecture.activation.ActivationFunction;
import com.github.neuralnetworks.architecture.activation.RepeaterFunction;
import com.github.neuralnetworks.architecture.input.ConstantInput;
import com.github.neuralnetworks.architecture.input.InputFunction;
import com.github.neuralnetworks.architecture.input.WeightedSum;
import com.github.neuralnetworks.architecture.neuron.BiasNeuron;
import com.github.neuralnetworks.util.Constants;

/**
 * a Multi Layer perceptron network
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
		} else if (!properties.containsKey(Constants.ACTIVATION_FUNCTION) || !properties.containsKey(Constants.INPUT_FUNCTION) || !properties.containsKey(Constants.ADD_BIAS)) {
			throw new IllegalArgumentException("Required properties not specified");
		}

		InputFunction inputFunction = (InputFunction) properties.get(Constants.INPUT_FUNCTION);
		ActivationFunction activationFunction = (ActivationFunction) properties.get(Constants.ACTIVATION_FUNCTION);
	    Boolean addBias = (Boolean) properties.get(Constants.ADD_BIAS);

	    // empty layers are created
	    List<Neuron[]> layers = new ArrayList<Neuron[]>();
	    inputNeurons = new Neuron[layerProperties.get(0)];
	    layers.add(inputNeurons);

	    int biasLength = addBias ? 1 : 0;
	    for (int i = 1; i < layerProperties.size(); i++) {
			layers.add(new Neuron[layerProperties.get(i) + biasLength]);
		}
	    outputNeurons = layers.get(layers.size() - 1);

	    // layer connections
	    List<FullyConnected> connections = new ArrayList<FullyConnected>();
	    for (int i = 1; i < layers.size(); i++) {
	    	connections.add(new FullyConnected(layers.get(i - 1), layers.get(i)));
	    }

	    // populate layers
	    InputFunction inputNeuronsInput = inputFunction instanceof WeightedSum ? inputFunction : new WeightedSum();

	    // populate input layer
	    ActivationFunction inputNeuronsActivation = new RepeaterFunction();
	    Neuron[] inputLayer = getInputNeurons();
	    IConnections firstConnections = connections.get(0);
	    for (int i = 0; i < inputLayer.length; i++) {
	    	Neuron n = new Neuron(inputNeuronsInput, inputNeuronsActivation);
	    	n.insert(null, firstConnections);
	    }

	    // populate the rest of the layers
	    ConstantInput ci = new ConstantInput();

	    for (int i = 1; i < layers.size(); i++) {
	    	Neuron[] l = layers.get(i);
	    	IConnections inbound = null, outbound = null;
	    	for (IConnections c : connections) {
	    		if (c.getInputNeurons() == l) {
	    			outbound = c;
	    		} else if (c.getOutputNeurons() == l) {
	    			inbound = c;
	    		}
	    	}

	    	if (addBias) {
	    		l[0] = new BiasNeuron(ci, outbound);
	    	}

	    	for (int j = 1; j < l.length; j++) {
		    	Neuron n = new Neuron(inputFunction, activationFunction);
		    	n.insert(inbound, outbound);
		    	n.getInboundConnections();
		    	n.getOutboundConnections();
	    	}
	    }
	}
}
