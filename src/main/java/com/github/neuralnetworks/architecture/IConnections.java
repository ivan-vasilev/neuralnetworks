package com.github.neuralnetworks.architecture;

/**
 * the classes that implement this interface need to be able to return either inbound or outbound connections to the neuron
 * @author hok
 *
 */
public interface IConnections extends IINputOutputNeurons {
	public NeuronConnections getConnections(Neuron n);
}
