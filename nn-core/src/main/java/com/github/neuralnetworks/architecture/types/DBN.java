package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.List;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.util.UniqueList;

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
	    addNeuralNetwork(new RBM(currentOutputLayer, layer, addBias, addBias));
	} else {
	    addLayer(layer);
	}

	return this;
    }

    @Override
    protected Collection<Layer> getRelevantLayers(RBM nn) {
	List<Layer> result = new UniqueList<Layer>();
	result.addAll(nn.getLayers());
	result.remove(nn.getVisibleBiasConnections().getInputLayer());

	return result;
    }
}
