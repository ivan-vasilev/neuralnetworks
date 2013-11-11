package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;


/**
 * Stacked autoencoder
 */
public class StackedAutoencoder extends DNN {

    public StackedAutoencoder(Layer input) {
	super();
	addLayer(input);
    }

    public void addLevel(Layer hidden, Layer output, boolean addBias) {
	Layer input = getOutputLayer();
	if (input == null) {
	    throw new IllegalArgumentException("At least one layer must be added before adding levels");
	}

	if (input.getNeuronCount() != output.getNeuronCount()) {
	    throw new IllegalArgumentException("Input and output layers must have equal number of neurons");
	}

	addNeuralNetwork(NNFactory.mlp(new Layer[] {input,  hidden, output}, addBias));
    }
}
