package com.github.neuralnetworks.calculation.operations.aparapi;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.aparapi.random.XORShiftKernel;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;

/**
 * Random noise, the mask is saved
 */
public class AparapiNoiseMask implements TensorFunction
{
	private static final long serialVersionUID = 1L;

	private transient AparapiNoiseMaskKernel kernel;
	private Tensor mask;
	private final float corruptionLevel;
	private final float corruptedValue;

	public AparapiNoiseMask(float corruptionLevel, float corruptedValue)
	{
		super();
		this.corruptionLevel = corruptionLevel;
		this.corruptedValue = corruptedValue;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		if (kernel == null || kernel.inputOutput != inputOutput.getElements())
		{
			mask = TensorFactory.tensor(inputOutput.getDimensions());
			kernel = new AparapiNoiseMaskKernel(inputOutput, mask, inputOutput.getSize(), corruptionLevel, corruptedValue);
		}

		kernel.execute(inputOutput.getSize());
	}

	public Tensor getMask()
	{
		return mask;
	}

	private static class AparapiNoiseMaskKernel extends XORShiftKernel
	{
		private final float corruptionLevel;
		private final int inputStartIndex;
		private float[] inputOutput;
		private final int maskStartIndex;
		private float[] maskArray;
		private final float corruptedValue;

		public AparapiNoiseMaskKernel(Tensor inputOutput, Tensor mask, int maximumRange, float corruptionLevel, float corruptedValue)
		{
			super(maximumRange);
			this.corruptionLevel = corruptionLevel;
			this.corruptedValue = corruptedValue;
			this.inputOutput = inputOutput.getElements();
			this.inputStartIndex = inputOutput.getStartIndex();
			this.maskArray = mask.getElements();
			this.maskStartIndex = mask.getStartIndex();
		}

		@Override
		public void run()
		{
			int id = getGlobalId();

			if (random01() < corruptionLevel)
			{
				maskArray[maskStartIndex + id] = 0;
				inputOutput[inputStartIndex + id] = corruptedValue;
			} else
			{
				maskArray[maskStartIndex + id] = 1;
			}
		}
	}
}
