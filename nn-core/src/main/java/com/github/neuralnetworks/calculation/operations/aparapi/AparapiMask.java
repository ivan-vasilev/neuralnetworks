package com.github.neuralnetworks.calculation.operations.aparapi;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Apply mask to tensor
 */
public class AparapiMask implements TensorFunction
{
	private static final long serialVersionUID = 1L;

	private AparapiNoiseMask noiseMask;
	private AparapiMaskKernel maskKernel;

	public AparapiMask(AparapiNoiseMask noiseMask)
	{
		super();
		this.noiseMask = noiseMask;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		if (maskKernel == null)
		{
			maskKernel = new AparapiMaskKernel(inputOutput, noiseMask.getMask());
		}

		maskKernel.execute(inputOutput.getSize());
	}

	private static class AparapiMaskKernel extends Kernel
	{
		private int inputStartIndex;
		private float[] inputOutput;
		private int maskStartIndex;
		private float[] maskArray;

		public AparapiMaskKernel(Tensor inputOutput, Tensor mask)
		{
			super();
			this.inputStartIndex = inputOutput.getStartIndex();
			this.inputOutput = inputOutput.getElements();
			this.maskStartIndex = mask.getStartIndex();
			this.maskArray = mask.getElements();
		}

		@Override
		public void run()
		{
			int id = getGlobalId();
			inputOutput[inputStartIndex + id] = inputOutput[inputStartIndex + id] * maskArray[maskStartIndex + id];
		}
	}
}
