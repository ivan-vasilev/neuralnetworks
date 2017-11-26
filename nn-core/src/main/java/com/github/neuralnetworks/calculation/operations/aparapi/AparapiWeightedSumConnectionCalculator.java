package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Simple weighted sum connection calculator
 */
public class AparapiWeightedSumConnectionCalculator extends ConnectionCalculatorImpl
{

	private static final long serialVersionUID = 5869298546838843306L;

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new AparapiWeightedSum(inputConnections.get(0), valuesProvider, targetLayer);
	}
}
