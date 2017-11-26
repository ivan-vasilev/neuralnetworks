package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.tensor.ValuesProvider;

public class OpenCLAveragePooling2DConnectionCalculator extends ConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1L;

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new OpenCLAveragePooling2D();
	}
}
