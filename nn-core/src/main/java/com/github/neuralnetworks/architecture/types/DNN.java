package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Default implementation of the DeepNeuralNetwork interface
 */
public class DNN extends NeuralNetworkImpl implements DeepNeuralNetwork {

    /**
     * List of networks in the network
     */
    private List<NeuralNetwork> neuralNetworks;

    public DNN() {
	super();
	this.neuralNetworks = new UniqueList<>();
    }

    @Override
    public List<NeuralNetwork> getNeuralNetworks() {
	return neuralNetworks;
    }

    /**
     * When a new network is added all it's layers are also added to the deep network
     * @param nn
     */
    protected void addNeuralNetwork(NeuralNetwork nn) {
	neuralNetworks.add(nn);
	if (nn.getLayers() != null) {
	    for (Layer l : nn.getLayers()) {
		addLayer(l);
	    }
	}
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.architecture.NeuralNetwork#getOutputLayer()
     * The output layer is the output layer of the last "child" network
     */
    @Override
    public Layer getOutputLayer() {
	NeuralNetwork nn = getLastNeuralNetwork();
	if (nn != null) {
	    return nn.getOutputLayer();
	}

	return null;
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.architecture.NeuralNetwork#getDataOutputLayer()
     * The data output layer is the data output layer of the last "child" network
     */
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
