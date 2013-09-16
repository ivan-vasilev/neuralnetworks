package com.github.neuralnetworks.architecture;

/**
 * the classes that implement this interface need to be able to return either inbound or outbound connections to the neuron
 *
 */
public interface IConnections extends IINputOutputLayers {

	/**
	 * @return - the list of input layer neurons that are participating in this connection graph
	 */
	public int[] getInputLayerNeurons();

	/**
	 * @return - the list of output layer neurons that are participating in this connection graph
	 */
	public int[] getOutputLayerNeurons();

	/**
	 * @return ConnectionGraph for feed-forward network
	 */
	public ConnectionGraph getForwardConnectionGraph();

	/**
	 * @return ConnectionGraph for feed-backward network
	 */
	public ConnectionGraph getBackwardConnectionGraph();
}
