package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * Autoencoder
 */
public class Autoencoder extends NeuralNetworkImpl {

    private Layer hiddenLayer;
    private Layer outputLayer;

    public Autoencoder(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, boolean addBias) {
	this.hiddenLayer = hiddenLayer;
	this.outputLayer = outputLayer;

	addLayer(inputLayer);
	addLayer(hiddenLayer);
	addLayer(outputLayer);

	new FullyConnected(inputLayer, outputLayer);
	new FullyConnected(inputLayer, outputLayer);

	if (addBias) {
	    Layer hiddenBiasLayer = new Layer(1, new ConstantConnectionCalculator(1));
	    addLayer(hiddenBiasLayer);
	    new FullyConnected(hiddenBiasLayer, hiddenLayer);

	    Layer outputBiasLayer = new Layer(1, new ConstantConnectionCalculator(1));
	    addLayer(outputBiasLayer);
	    new FullyConnected(outputBiasLayer, outputLayer);
	}
    }

    @Override
    public Layer getOutputLayer() {
	return hiddenLayer;
    }

    @Override
    public Layer getDataOutputLayer() {
	return outputLayer;
    }
}
