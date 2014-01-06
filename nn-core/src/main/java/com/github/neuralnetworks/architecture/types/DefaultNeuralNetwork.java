package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

/**
 * Default implementation of the network
 */
public class DefaultNeuralNetwork extends NeuralNetworkImpl {

    @Override
    public Layer getOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }

    @Override
    public Layer getDataOutputLayer() {
	return getNoOutboundConnectionsLayer();
    }
}
