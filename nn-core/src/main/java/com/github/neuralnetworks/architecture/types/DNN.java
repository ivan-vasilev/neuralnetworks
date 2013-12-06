package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.DeepNeuralNetwork;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Deep neural network
 */
public class DNN extends NeuralNetworkImpl implements DeepNeuralNetwork {

    private List<NeuralNetwork> neuralNetworks;

    public DNN() {
	super();
	this.neuralNetworks = new UniqueList<>();
    }

    @Override
    public List<NeuralNetwork> getNeuralNetworks() {
	return neuralNetworks;
    }

    protected void addNeuralNetwork(NeuralNetwork nn) {
	neuralNetworks.add(nn);
	if (nn.getLayers() != null) {
	    for (Layer l : nn.getLayers()) {
		addLayer(l);
	    }
	}
    }

    @Override
    public Layer getOutputLayer() {
	NeuralNetwork nn = getLastNeuralNetwork();
	if (nn != null) {
	    return nn.getOutputLayer();
	}

	return null;
    }

    @Override
    public Layer getDataOutputLayer() {
	NeuralNetwork nn = getLastNeuralNetwork();
	if (nn != null) {
	    return nn.getDataOutputLayer();
	}

	return null;
    }

    public NeuralNetwork getFirstNeuralNetwork() {
	if (neuralNetworks != null && neuralNetworks.size() > 0) {
	    return neuralNetworks.get(0);
	}

	return null;
    }

    public NeuralNetwork getLastNeuralNetwork() {
	if (neuralNetworks != null && neuralNetworks.size() > 0) {
	    return neuralNetworks.get(neuralNetworks.size() - 1);
	}
	
	return null;
    }
}
