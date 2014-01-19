package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.HashSet;

import com.github.neuralnetworks.architecture.Layer;

/**
 * Deep Belief Network
 */
public class DBN extends DNN<RBM> {

    public DBN() {
	super();
    }

    /**
     * For each added layer a new RBM is created with visible layer - the hidden layer of the previous network and hidden layer - the new layer
     * @param layer
     * @param addBias
     * @return this
     */
    public DBN addLevel(Layer layer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (currentOutputLayer != null) {
	    addNeuralNetwork(new RBM(currentOutputLayer, layer, false, addBias));
	} else {
	    addLayer(layer);
	}

	return this;
    }

    @Override
    protected Collection<Layer> getRelevantLayers(RBM nn) {
	Collection<Layer> result = null;
	if (getNeuralNetworks().size() == 0 && nn.getVisibleBiasConnections() != null) {
	    result = new HashSet<Layer>();
	    result.addAll(nn.getLayers());
	    result.remove(nn.getVisibleBiasConnections().getInputLayer());
	} else {
	    result = nn.getLayers();
	}

	return result;
    }
}
