package com.github.neuralnetworks.architecture.types;

import java.util.Collection;
import java.util.List;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Deep Belief Network
 */
public class DBN extends DNN<RBM>
{

	private static final long serialVersionUID = 1L;

	public DBN()
	{
		super();
	}

	@Override
	protected Collection<Layer> getRelevantLayers(RBM nn)
	{
		List<Layer> result = new UniqueList<Layer>();
		result.addAll(nn.getLayers());
		if (nn.getVisibleBiasConnections() != null)
		{
			result.remove(nn.getVisibleBiasConnections().getInputLayer());
		}

		return result;
	}
}
