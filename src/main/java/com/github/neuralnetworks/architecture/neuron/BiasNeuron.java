package com.github.neuralnetworks.architecture.neuron;

import com.github.neuralnetworks.architecture.IConnections;
import com.github.neuralnetworks.architecture.Neuron;
import com.github.neuralnetworks.architecture.input.ConstantInput;

/**
 *
 * Bias neuron
 *
 */
public class BiasNeuron extends Neuron {
	public BiasNeuron(ConstantInput inputFunction, IConnections outboundConnections) {
		super(inputFunction, null, null, outboundConnections);
	}
}
