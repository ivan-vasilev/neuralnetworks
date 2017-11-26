package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * based on the idea that the input layer could be another one than the last added one, the name of the input layer can (still optionally) be used as parameter
 *
 * @author tmey
 */
public abstract class NamedSingleInputLayerBuilder extends NamedLayerBuilder
{

	private String inputLayerName = null;

	public NamedSingleInputLayerBuilder(String default_layer_name)
	{
		super(default_layer_name);
	}

	@Override
	public Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Hyperparameters hyperparameters)
	{
		return build(neuralNetwork, newLayerName, getInputLayer(neuralNetwork), hyperparameters);
	}

	protected abstract Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters);

	public Layer getInputLayer(NeuralNetworkImpl neuralNetwork)
	{
		if (inputLayerName != null)
		{
			for (Layer layer : neuralNetwork.getLayers())
			{
				if (inputLayerName.equals(layer.getName()))
				{
					return layer;
				}
			}
			throw new IllegalArgumentException("Can't find the input layer with name " + inputLayerName);
		} else
		{
			return neuralNetwork.getOutputLayer();
		}
	}


	public String getInputLayerName()
	{
		return inputLayerName;
	}

	public void setInputLayerName(String inputLayerName)
	{
		this.inputLayerName = inputLayerName;
	}
}
