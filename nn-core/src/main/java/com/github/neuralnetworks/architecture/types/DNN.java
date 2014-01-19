package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.List;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Default implementation of the DeepNeuralNetwork interface
 */
public abstract class DNN<N extends NeuralNetwork> extends NeuralNetworkImpl implements DeepNeuralNetwork<N> {

    /**
     * List of networks in the network
     */
    private List<N> neuralNetworks;

    public DNN() {
	super();
	this.neuralNetworks = new UniqueList<>();
    }

    @Override
    public List<N> getNeuralNetworks() {
	return neuralNetworks;
    }

    /**
     * When a new network is added all it's layers are also added to the deep network
     * @param nn
     */
    protected void addNeuralNetwork(N nn) {
	addLayers(getRelevantLayers(nn));
	neuralNetworks.add(nn);
    };

    /**
     * Retrieves the layers that are relevant in the context of the deep network
     * @param nn
     */
    protected abstract Collection<Layer> getRelevantLayers(N nn);

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
