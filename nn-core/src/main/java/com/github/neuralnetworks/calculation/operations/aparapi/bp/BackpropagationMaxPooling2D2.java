package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Properties;

/**
 * Backpropagation for max pooling layers
 */
public class BackpropagationMaxPooling2D2 extends BackPropagationConnectionCalculatorImpl
{
	private static final long serialVersionUID = 8165829315701496713L;

	public BackpropagationMaxPooling2D2(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		Subsampling2DConnection con = null;
		for (Connections c : inputConnections)
		{
			if (c instanceof Subsampling2DConnection)
			{
				con = (Subsampling2DConnection) c;
				break;
			}
		}

		if (con != null)
		{
			connectionCalculators.put(con, new BackpropagationMaxPooling2DCC(con, valuesProvider, activations));
		}
	}

	public static class BackpropagationMaxPooling2DCC extends AparapiBackpropagationSubsampling2D2
	{
		private static final long serialVersionUID = -8888670594631428090L;

		// protected Subsampling2DConnection connection;

		public BackpropagationMaxPooling2DCC(Subsampling2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations)
		{
			super(c, valuesProvider, activations);
			// this.connection = c;
		}

		@Override
		protected void pool(int inputFilterStartIndex, int outputIndex, int sample)
		{
			int id = getGlobalId();
			boolean isMax = true;
			float max = ffActivation[inputStartIndex + id + sample * inputMiniBatchDistance];
			int inputStart = inputFilterStartIndex + sample * inputMiniBatchDistance;
			for (int j = 0; j < regionLength && isMax; j++)
			{
				if (ffActivation[inputStart + featureMapOffsets[j]] > max)
				{
					isMax = false;
				}

//					CustomArray ca = new CustomArray(new int[] {
//							inputStart + featureMapOffsets[j],
//							outputIndex + i * outputMiniBatchDistance
//					});
//					if (!CustomArrays.getInstance().getCustomArrays().get(connection).contains(ca))
//					{
//						throw new RuntimeException(Arrays.toString(ca.getArray()));
//					}
			}

			if (isMax)
			{
				input[inputStartIndex + id + sample * inputMiniBatchDistance] += output[outputIndex + sample * outputMiniBatchDistance];
			}
		}
	}
}