package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * @author tmey
 */
public interface LayerBuilder
{

	public default Layer build(NeuralNetworkImpl neuralNetwork)
    {
        return build(neuralNetwork, null);
    }

    public Layer build(NeuralNetworkImpl neuralNetwork,Hyperparameters hyperparameters);

}
