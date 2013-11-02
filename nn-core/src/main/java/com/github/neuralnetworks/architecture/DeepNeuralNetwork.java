package com.github.neuralnetworks.architecture;

import java.util.List;

public interface DeepNeuralNetwork extends NeuralNetwork {
    public List<NeuralNetwork> getNeuralNetworks();
}
