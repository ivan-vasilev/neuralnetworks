package com.github.neuralnetworks.calculation.operations.cpu;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Softmax activation function
 */
public class SoftmaxFunction extends Kernel implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	private float[] values;
	private int startIndex;
	private int nextRowStep;
	private int nextColumnStep;
	private int columns;

	@Override
	public void value(Tensor inputOutput)
	{
		Matrix io = (Matrix) inputOutput;

		this.values = io.getElements();
		this.startIndex = io.getStartIndex();
		this.nextRowStep = io.getRowElementsDistance();
		this.nextColumnStep = io.getColumnElementsDistance();
		this.columns = io.getColumns();

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, io.getRows());
	}

	@Override
	public void run()
	{
		float sum = 0;
		int start = startIndex + getGlobalId() * nextRowStep;
		int c = columns;

		float max = values[start];
		for (int i = 1; i < c; i++)
		{
			max = max(max, values[start + i * nextColumnStep]);
		}

		for (int i = 0; i < c; i++)
		{
			sum += exp(values[start + i * nextColumnStep] - max);
		}

		for (int i = 0; i < c; i++)
		{
			values[start + i * nextColumnStep] = exp(values[start + i * nextColumnStep] - max) / sum;
		}
	}
}
