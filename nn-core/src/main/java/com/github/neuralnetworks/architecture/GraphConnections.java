package com.github.neuralnetworks.architecture;

/**
 * the classes that implement this interface need to be able to return either
 * inbound or outbound connections to the neuron
 * 
 */
public interface GraphConnections extends Connections {

    /**
     * @return the start neuron of the input layer neurons
     */
    public int getInputLayerStartNeuron();

    /**
     * @return the start neuron of the output layer neurons
     */
    public int getOutputLayerStartNeuron();

    /**
     * @return ConnectionGraph with weights
     */
    public Matrix getConnectionGraph();
}
