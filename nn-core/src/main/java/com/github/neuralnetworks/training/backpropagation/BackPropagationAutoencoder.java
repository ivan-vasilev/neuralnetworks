package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * BackPropagation for autoencoders (input and target are the same). Supports
 * denoising autoencoders.
 */
public class BackPropagationAutoencoder extends BackPropagationTrainer<Autoencoder>
{

	private static final long serialVersionUID = 1L;

	public BackPropagationAutoencoder(Properties properties)
	{
		super(properties);
		setTrainingInputProvider(new AutoencoderTrainingInputrovider(getTrainingInputProvider(), getCorruptionRate()));
	}

	public Float getCorruptionRate()
	{
		return getProperties().getParameter(Constants.CORRUPTION_LEVEL);
	}

	public void setCorruptionRate(Float corruptionRate)
	{
		getProperties().setParameter(Constants.CORRUPTION_LEVEL, corruptionRate);
	}

	private static class AutoencoderTrainingInputrovider extends TrainingInputProviderImpl
	{

		private static final long serialVersionUID = 1L;

		private TrainingInputProvider base;
		private Float corruptionRate;
		private TensorFunction noise;
		private Tensor input;

		public AutoencoderTrainingInputrovider(TrainingInputProvider base, Float corruptionRate)
		{
			super();
			this.base = base;
			this.corruptionRate = corruptionRate;
		}

		@Override
		public int getInputSize()
		{
			return base.getInputSize();
		}

		@Override
		public void getNextInput(Tensor t)
		{
			this.input = t;
			base.getNextInput(t);
		}

		@Override
		public void getNextTarget(Tensor target)
		{
			if (corruptionRate != null && corruptionRate > 0)
			{
				if (noise == null)
				{
					noise = OperationsFactory.noise(target, corruptionRate, 0);
				}

				System.arraycopy(input.getElements(), input.getStartIndex(), target.getElements(), target.getStartIndex(), input.getSize());
				noise.value(target);
			} else
			{
				target.setElements(input.getElements());
			}
		}

		@Override
		public void beforeBatch(TrainingInputData ti)
		{
			super.beforeBatch(ti);
			base.beforeBatch(ti);
		}

		@Override
		public void afterBatch(TrainingInputData ti)
		{
			super.afterBatch(ti);
			base.afterBatch(ti);
		}

		@Override
		public void reset()
		{
			super.reset();
			base.reset();
		}

		@Override
		public int getInputDimensions()
		{
			return base.getInputDimensions();
		}

		@Override
		public int getTargetDimensions()
		{
			return base.getInputDimensions();
		}
	}
}
