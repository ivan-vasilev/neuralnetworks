package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * @author tmey
 */
public class DropoutLayerBuilder extends NamedSingleInputLayerBuilder
{

	public static String DEFAULT_LAYER_NAME = "DropoutLayer";

	public float rate = 0.5f;

	public DropoutLayerBuilder()
	{
		super(DEFAULT_LAYER_NAME);
	}

	@Override
	protected Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{

		// TODO write content

		return null;
	}

	public float getRate()
	{
		return rate;
	}

	public void setRate(float rate)
	{

		if (rate <= 0)
		{
			throw new IllegalArgumentException(
					"The drop rate must be larger than zero! (current value: " + rate + ")");
		}

		this.rate = rate;
	}

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder("DropoutLayerBuilder{");
		sb.append("name=").append(getName());
		sb.append(", inputLayer=").append(getInputLayerName());
		sb.append(", dropoutRate=").append(getRate());
		sb.append('}');
		return sb.toString();
	}
}
