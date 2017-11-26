package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Average pooling
 */

public class AparapiAveragePooling2D extends ConnectionCalculatorImpl
{

	private static final long serialVersionUID = 8165829315701496713L;

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new AparapiAveragePooling2DCC((Subsampling2DConnection) inputConnections.get(0), valuesProvider, targetLayer);
	}

	public static class AparapiAveragePooling2DCC extends AparapiSubsampling2D
	{

		private static final long serialVersionUID = -2393526660090301257L;

		public AparapiAveragePooling2DCC(Subsampling2DConnection c, ValuesProvider valuesProvider, Layer targetLayer)
		{
			super(c, valuesProvider, targetLayer);
		}

		@Override
		protected void pool(int inputStartIndex, int outputStartIndex)
		{
			float sum = 0;
			int inputStart = 0;
			for (int i = 0; i < miniBatchSize; i++)
			{
				sum = 0;
				inputStart = inputStartIndex + i * inputMiniBatchDistance;
				for (int j = 0; j < regionLength; j++)
				{
					sum += input[inputStart + featureMapOffsets[j]];
				}

				output[outputStartIndex + i * outputMiniBatchDistance] = sum / regionLength;
			}
		}
	}
}