package com.github.neuralnetworks.calculation.operations.aparapi;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Base class for all feedforward convolutional functions
 */
public class AparapiConv2DFF extends AparapiConv2D
{

	private static final long serialVersionUID = 5048904661076337615L;

	private Conv2DConnection connection;

	private int clear;

	public AparapiConv2DFF(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super(c, valuesProvider, targetLayer);
		this.connection = c;

		Tensor o = TensorFactory.tensor(c.getOutputLayer(), c, valuesProvider);

		if (!ClearValuesManager.getInstance().isCleared(o))
		{
			ClearValuesManager.getInstance().addToCleared(o);
			clear = 0;
		} else
		{
			clear = 1;
		}

//		CustomArrays.getInstance().getCustomArrays().put(connection, new HashSet<>());
	}

	@Override
	protected void conv(int weightsStartId, int inputStartId, int outputStartId)
	{
		// calculate sum based on feature map offsets and feature map weights
		float sum = 0;
		int inputStart = 0;
		for (int i = 0; i < miniBatchSize; i++)
		{
			inputStart = i * inputMiniBatchDistance + inputStartId;
			sum = output[outputStartId + i * outputMiniBatchDistance] * clear;

			for (int j = 0; j < featureMapWeights; j++)
			{
				sum += input[inputStart + featureMapOffsets[j]] * weights[weightsStartId + j];
//				CustomArray ca = new CustomArray(new int[] { inputStart + featureMapOffsets[j], weightsStartId + j, outputStartId + i * outputMiniBatchDistance});
//				CustomArrays.getInstance().getCustomArrays().get(connection).add(ca);
			}

			output[outputStartId + i * outputMiniBatchDistance] = sum;
		}
	}

	public Conv2DConnection getConnection()
	{
		return connection;
	}

	public int getClear()
	{
		return clear;
	}
}
