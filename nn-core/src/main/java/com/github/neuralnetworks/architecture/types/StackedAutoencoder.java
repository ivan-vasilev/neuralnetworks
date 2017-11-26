package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;

/**
 * Stacked autoencoder
 */
public class StackedAutoencoder extends DNN<Autoencoder>
{

	private static final long serialVersionUID = 1L;

	public StackedAutoencoder(Layer input)
	{
		super();
		addLayer(input);
	}

	@Override
	protected Collection<Layer> getRelevantLayers(Autoencoder nn)
	{
		Set<Layer> layers = new HashSet<Layer>();
		layers.add(nn.getHiddenLayer());
		layers.add(nn.getInputLayer());

		if (nn.getHiddenBiasLayer() != null)
		{
			layers.add(nn.getHiddenBiasLayer());
		}

		return layers;
	}
}
