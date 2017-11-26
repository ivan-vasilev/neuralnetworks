package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * If you want to give your layer a unique name or automatic create one, than use this class and the function getLayerName()
 *
 * @author tmey
 */
public abstract class NamedLayerBuilder implements LayerBuilder
{

	protected final String DEFAULT_LAYER_NAME;
	private String name;

	public NamedLayerBuilder(String default_layer_name)
	{
		DEFAULT_LAYER_NAME = default_layer_name;
	}


	protected String getLayerName(NeuralNetworkImpl neuralNetwork)
	{
		return getLayerName(neuralNetwork, DEFAULT_LAYER_NAME);
	}

	public abstract Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Hyperparameters hyperparameters);

	@Override
	public Layer build(NeuralNetworkImpl neuralNetwork, Hyperparameters hyperparameters)
	{
		return build(neuralNetwork, getLayerName(neuralNetwork), hyperparameters);
	}

	/**
	 * @param neuralNetwork
	 * @return
	 */
	protected String getLayerName(NeuralNetworkImpl neuralNetwork, String defaultName)
	{
		if (defaultName == null)
		{
			throw new IllegalArgumentException("defaultName must be not null!");
		}


		if (name != null)
		{
			// test if unique

			for (Layer l : neuralNetwork.getLayers())
			{
				if (l.getName().equals(name))
				{
					throw new IllegalArgumentException("the layer name " + name + " is already used in the network!");
				}
			}

			return name;
		} else
		{
			// find free layer name

			String finalLayerName = defaultName;
			nameLoop: for (int i = 1;; i++)
			{

				if (i != 1)
				{
					finalLayerName = defaultName + "_" + i;
				}

				for (Layer l : neuralNetwork.getLayers())
				{
					if (l.getName().equals(finalLayerName))
					{
						continue nameLoop;
					}
				}
				break;
			}

			return finalLayerName;
		}


	}

	public String getName()
	{
		return name;
	}

	public void setName(String name)
	{
		this.name = name;
	}
}
