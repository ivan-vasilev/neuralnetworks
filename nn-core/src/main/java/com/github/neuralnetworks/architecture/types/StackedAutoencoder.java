package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;

/**
 * Stacked autoencoder
 */
public class StackedAutoencoder extends DNN<Autoencoder> {

    public StackedAutoencoder(Layer input) {
	super();
	addLayer(input);
    }

    /**
     * This method creates new Autoencoder with input layer - the hidden layer of the previous topmost autoencoder.
     */
    public StackedAutoencoder addLevel(Layer layer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (currentOutputLayer != null) {
	    addNeuralNetwork(new Autoencoder(currentOutputLayer, layer, new Layer(currentOutputLayer.getNeuronCount()), addBias));
	} else {
	    addLayer(layer);
	}

	return this;
    }

    @Override
    protected Collection<Layer> getRelevantLayers(Autoencoder nn) {
	Set<Layer> layers = new HashSet<Layer>();
	layers.add(nn.getHiddenLayer());
	layers.add(nn.getInputLayer());

	if (nn.getHiddenBiasLayer() != null) {
	    layers.add(nn.getHiddenBiasLayer());
	}

	return layers;
    }
}
