package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Default implementation of Connection calculator for convolutional/subsampling layers
 */
public class ConnectionCalculatorConv extends ConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1L;

	public ConnectionCalculatorConv()
	{
		super();
	}

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new AparapiConv2DFF((Conv2DConnection) inputConnections.get(0), valuesProvider, targetLayer);
	}
}
