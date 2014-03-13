package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.util.Matrix;

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
