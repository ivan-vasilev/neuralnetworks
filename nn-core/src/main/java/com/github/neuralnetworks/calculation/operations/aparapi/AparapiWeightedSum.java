package com.github.neuralnetworks.calculation.operations.aparapi;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Base Aparapi connection calculator for weighted sum functions (matrix
 * multiplication). If there are multiple inbound connections they are combined
 * in a "single" connection and are calculated simultaneously
 * 
 * !!! IMPORTANT !!! Aparapi only works one-dimensional arrays of primitive data
 * types can only call member methods of the Kernel class itself.
 * 
 * Because of this limitations all the data that is contained in the input
 * connections, weight matrices, input values etc is converted into
 * one-dimensional member arrays of this class
 */
public class AparapiWeightedSum extends AparapiFullyConnected
{

	private static final long serialVersionUID = 1L;

	private int clear;

	public AparapiWeightedSum(Connections inputConnection, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super(inputConnection, valuesProvider, targetLayer);
		Matrix o = TensorFactory.tensor(targetLayer, inputConnection, valuesProvider);

		if (!ClearValuesManager.getInstance().isCleared(o))
		{
			ClearValuesManager.getInstance().addToCleared(o);
			clear = 0;
		} else
		{
			clear = 1;
		}
	}

	@Override
	public void run()
	{
		int id = getGlobalId();

		float value = 0;
		int inputStartIndex = 0;
		int outputStartIndex = outputStartPosition + id * outputColumnStep;
		int weightStartIndex = weightStartPosition + weightsInitialStep * id;

		// each input example
		for (int i = 0; i < miniBatchSize; i++)
		{
			// each connection (of the combined connections)
			value = output[outputStartIndex + i * outputRowStep] * clear;
			// each element in the row/column

			inputStartIndex = inputStartPosition + i * inputRowStep;
			for (int j = 0; j < weightsSize; j++)
			{
				value += input[inputStartIndex + j * inputColumnStep] * weights[weightStartIndex + j * weightsStep];
			}

			output[outputStartIndex + i * outputRowStep] = value;
		}

		after();
	}

	protected void after()
	{
	}

	public int getClear()
	{
		return clear;
	}
}
