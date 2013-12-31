package com.github.neuralnetworks.architecture;

/**
 * Classes that implement this interface need to be able to return either
 * inbound or outbound connections to the neuron
 */
public interface GraphConnections extends Connections {

    /**
     * @return ConnectionGraph with weights
     */
    public Matrix getConnectionGraph();
}
