package com.github.neuralnetworks.architecture;

/**
 * Default interface for connections between the layers Each Connection is also
 * a NeuralNetwork of itself
 */
public interface Connections extends NeuralNetwork {
    public int getInputUnitCount();

    public int getOutputUnitCount();
}
