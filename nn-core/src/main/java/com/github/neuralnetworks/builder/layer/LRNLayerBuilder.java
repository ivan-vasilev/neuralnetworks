package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * @author tmey
 */
public class LRNLayerBuilder extends NamedSingleInputLayerBuilder
{

	public static String DEFAULT_LAYER_NAME = "LRNLayer";

	public LRNLayerBuilder()
	{
		super(DEFAULT_LAYER_NAME);
	}

	@Override
	protected Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{

		// TODO write content

		return null;
	}
}