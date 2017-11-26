package com.github.neuralnetworks.calculation.operations.opencl.kernels;

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
 * Convolutional Backpropagation connection calculator without activation function (for connections to the output layer)
 */
public class OpenCLWeightedSumBP extends BackPropagationConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1178188233641224762L;

	public OpenCLWeightedSumBP(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activatinos, Layer targetLayer)
	{
		for (Connections c : inputConnections)
		{
			if (Util.isBias(c.getInputLayer()) && targetLayer != c.getInputLayer())
			{
				connectionCalculators.put(c, new OpenCLWeightedSumBPCC());
			} else
			{
				connectionCalculators.put(c, new OpenCLWeightedSumBPCC());
			}
		}
	}

	/**
	 * just extends OpenCLWeightSum without changing anything, i.e. is tested via OpenCLWeightSum
	 */
	public static class OpenCLWeightedSumBPCC extends OpenCLWeightedSum implements BackPropagationConnectionCalculator
	{
		private static final long serialVersionUID = 1L;

		@Override
		public ValuesProvider getActivations()
		{
			return null;
		}

		@Override
		public void setActivations(ValuesProvider activations)
		{
		}
	}
}
