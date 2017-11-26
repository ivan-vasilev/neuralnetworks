package com.github.neuralnetworks.calculation.operations.aparapi.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Bernoulli distribution
 */
public class BernoulliDistribution implements TensorFunction
{

	private static final long serialVersionUID = 1L;

	protected Map<Integer, BernoulliKernel> kernels = new HashMap<>();

	@Override
	public void value(Tensor inputOutput)
	{
		BernoulliKernel kernel = kernels.get(inputOutput.getElements().length);
		if (kernel == null)
		{
			kernels.put(inputOutput.getElements().length, kernel = new BernoulliKernel(inputOutput.getElements().length));
		}

		kernel.values = inputOutput.getElements();

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(kernel, inputOutput.getSize());
	}

	private static class BernoulliKernel extends XORShiftKernel
	{

		private float[] values;

		public BernoulliKernel(int maximumRange)
		{
			super(maximumRange);
		}

		@Override
		public void run()
		{
			int id = getGlobalId();
			if (values[id] > random01())
			{
				values[id] = 1;
			} else
			{
				values[id] = 0;
			}
		}
	}
}
