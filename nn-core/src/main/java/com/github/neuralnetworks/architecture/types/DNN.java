package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.DeepNeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Deep neural network
 */
public class DNN extends NeuralNetworkImpl implements DeepNeuralNetwork {

    protected List<NeuralNetwork> neuralNetworks;

    public DNN() {
	super();
	this.neuralNetworks = new UniqueList<>();
    }

    @Override
    public List<NeuralNetwork> getNeuralNetworks() {
	return neuralNetworks;
    }
}
