package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;

/**
 * a Multi Layer perceptron network
 * 
 */
public class MultiLayerPerceptron extends NeuralNetworkImpl {
    public MultiLayerPerceptron addLayer(Layer layer, boolean addBias) {
	if (addLayer(layer) && getOutputLayer() != layer) {
	    new FullyConnected(getOutputLayer(), layer);
	}

	if (addBias) {
	    Layer biasLayer = new Layer(layer.getNeuronCount(), new ConstantConnectionCalculator(1));
	    addLayer(biasLayer);
	    new OneToOne(biasLayer, layer);
	}

	return this;
    }
}
