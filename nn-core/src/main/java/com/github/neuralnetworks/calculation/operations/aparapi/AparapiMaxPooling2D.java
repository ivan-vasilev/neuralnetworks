package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D extends ConnectionCalculatorImpl
{

	private static final long serialVersionUID = 8165829315701496713L;

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new AparapiMaxPooling2DCC((Subsampling2DConnection) inputConnections.get(0), valuesProvider, targetLayer);
	}

	public static class AparapiMaxPooling2DCC extends AparapiSubsampling2D
	{

		private static final long serialVersionUID = -2393526660090301257L;

		//private Subsampling2DConnection connection;

		public AparapiMaxPooling2DCC(Subsampling2DConnection c, ValuesProvider valuesProvider, Layer targetLayer)
		{
			super(c, valuesProvider, targetLayer);
			//this.connection = c;
			//CustomArrays.getInstance().getCustomArrays().put(connection, new HashSet<>());
		}

		@Override
		protected void pool(int inputStartIndex, int outputStartIndex)
		{
			float max = 0;
			int inputStart = 0;
			for (int i = 0; i < miniBatchSize; i++)
			{
				inputStart = inputStartIndex + i * inputMiniBatchDistance;
				max = input[inputStart + featureMapOffsets[0]];

//				CustomArray ca = new CustomArray(new int[] { inputStart + featureMapOffsets[0], outputStartIndex + i * outputMiniBatchDistance});
//				CustomArrays.getInstance().getCustomArrays().get(connection).add(ca);

				for (int j = 1; j < regionLength; j++)
				{
					max = max(input[inputStart + featureMapOffsets[j]], max);

//					ca = new CustomArray(new int[] { inputStart + featureMapOffsets[j], outputStartIndex + i * outputMiniBatchDistance});
//					CustomArrays.getInstance().getCustomArrays().get(connection).add(ca);
				}

				output[outputStartIndex + i * outputMiniBatchDistance] = max;
			}
		}
	}
}
