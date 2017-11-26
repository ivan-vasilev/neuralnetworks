package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Backpropagation connection calculator for fully connected input layers
 */
public class BackPropagationFullyConnected extends BackPropagationConnectionCalculatorImpl
{

	private static final long serialVersionUID = 1178188233641224762L;

	public BackPropagationFullyConnected(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		for (Connections c : inputConnections)
		{
			if (Util.isBias(c.getInputLayer()) && targetLayer != c.getInputLayer())
			{
				connectionCalculators.put(c, new AparapiBackpropagationFullyConnected(c, valuesProvider, activations, getWeightUpdates().get(c), c.getInputLayer()));
			} else
			{
				connectionCalculators.put(c, new AparapiBackpropagationFullyConnected(c, valuesProvider, activations, getWeightUpdates().get(c), targetLayer));
			}
		}
	}
}
