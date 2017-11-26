package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.cpu.MaxoutWinners;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Maxout activation
 */
public class AparapiMaxout extends ConnectionCalculatorImpl
{

	private static final long serialVersionUID = -6602713983386107132L;

	@Override
	protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		return new AparapiMaxoutFunction(inputConnections.get(0), valuesProvider, targetLayer);
	}

	public static class AparapiMaxoutFunction extends AparapiFullyConnected
	{

		private static final long serialVersionUID = 2572354641295173835L;

		private final int winnersStartPosition;
		private final int[] maxoutWinners;

		public AparapiMaxoutFunction(Connections inputConnection, ValuesProvider valuesProvider, Layer targetLayer)
		{
			super(inputConnection, valuesProvider, targetLayer);
			MaxoutWinners.getInstance().setBatchSize(TensorFactory.batchSize(valuesProvider));
			winnersStartPosition = MaxoutWinners.getInstance().getStartPositions(inputConnection);
			maxoutWinners = MaxoutWinners.getInstance().getWinners();
		}

		@Override
		public void run()
		{
			int id = getGlobalId();

			int maxIndex = 0;
			float max = 0, current = 0;

			// each input example
			for (int i = 0; i < miniBatchSize; i++)
			{
				// each element in the row/column

				maxIndex = 0;
				current = max = input[inputStartPosition + i * inputColumnStep] * weights[weightStartPosition];
				for (int j = 1; j < weightsSize; j++)
				{
					current = input[inputStartPosition + j * inputRowStep + i * inputColumnStep] * weights[weightStartPosition + weightsInitialStep * id + j * weightsStep];
					if (current > max)
					{
						max = current;
						maxIndex = j;
					}
				}

				maxoutWinners[winnersStartPosition + id * miniBatchSize + i] = maxIndex;

				output[outputStartPosition + id * outputRowStep + i * outputColumnStep] += max;
			}
		}
	}
}
