package com.github.neuralnetworks.input;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;

public class MultipleNeuronsOutputError implements OutputError
{
	private static final long serialVersionUID = 1L;

	private List<OutputTargetTuple> tuples;
	private Map<Integer, Integer> outputToTarget;
	private int nullCount;
	private int dim;

	public MultipleNeuronsOutputError()
	{
		super();
		reset();
	}

	@Override
	public void addItem(Tensor newtorkOutput, Tensor targetOutput)
	{
		Matrix target = (Matrix) targetOutput;
		Matrix actual = (Matrix) newtorkOutput;

		if (dim == -1)
		{
			dim = target.getColumns();
		}

		if (!Arrays.equals(actual.getDimensions(), target.getDimensions()))
		{
			throw new IllegalArgumentException("Dimensions don't match");
		}

		for (int i = 0; i < target.getRows(); i++)
		{
			boolean hasDifferentValues = false;
			for (int j = 0; j < actual.getColumns(); j++)
			{
				if (actual.get(i, j) != actual.get(i, 0))
				{
					hasDifferentValues = true;
					break;
				}
			}

			if (hasDifferentValues)
			{
				int targetPos = 0;
				for (int j = 0; j < target.getColumns(); j++)
				{
					if (target.get(i, j) == 1)
					{
						targetPos = j;
						break;
					}
				}

				int outputPos = 0;
				float max = actual.get(i, 0);
				for (int j = 0; j < actual.getColumns(); j++)
				{
					if (actual.get(i, j) > max)
					{
						max = actual.get(i, j);
						outputPos = j;
					}
				}

				tuples.add(new OutputTargetTuple(outputPos, targetPos));
			} else
			{
				nullCount++;
			}
		}
	}

	@Override
	public float getTotalNetworkError()
	{
		return getTotalInputSize() > 0 ? ((float) getTotalErrorSamples()) / getTotalInputSize() : 0;
	}

	@Override
	public int getTotalErrorSamples()
	{
		if (outputToTarget == null)
		{
			outputToTarget = outputToTarget();
		}

		int errorSamples = 0;
		for (OutputTargetTuple t : tuples)
		{
			if (!outputToTarget.get(t.outputPos).equals(t.targetPos))
			{
				errorSamples++;
			}
		}

		return nullCount + errorSamples;
	}

	@Override
	public int getTotalInputSize()
	{
		return tuples.size() + nullCount;
	}

	private Map<Integer, Integer> outputToTarget()
	{
		Map<Integer, Integer> result = new HashMap<>();
		Map<Integer, int[]> targetToOutput = new HashMap<>();
		for (OutputTargetTuple t : tuples)
		{
			if (!targetToOutput.containsKey(t.targetPos))
			{
				targetToOutput.put(t.targetPos, new int[dim]);
			}

			targetToOutput.get(t.targetPos)[t.outputPos]++;
		}

		for (int i = 0; i < dim; i++)
		{
			int[] d = targetToOutput.get(i);
			if (d != null)
			{
				int max = 0;
				for (int j = 0; j < dim; j++)
				{
					if (d[j] > d[max] && !result.values().contains(j))
					{
						max = j;
					}
				}

				result.put(i, max);
			}
		}

		return result;
	}

	@Override
	public void reset()
	{
		this.tuples = new ArrayList<>();
		this.dim = -1;
		this.nullCount = 0;
		this.outputToTarget = null;
	}

	private static class OutputTargetTuple implements Serializable
	{
		private static final long serialVersionUID = 1L;

		public OutputTargetTuple(Integer outputPos, Integer targetPos)
		{
			this.outputPos = outputPos;
			this.targetPos = targetPos;
		}

		public Integer outputPos;
		public Integer targetPos;
	}
}
