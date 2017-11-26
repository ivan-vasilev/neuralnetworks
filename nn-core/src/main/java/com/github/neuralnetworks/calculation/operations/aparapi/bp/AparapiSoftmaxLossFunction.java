package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.util.Environment;

/**
 * Mean squared error derivative
 */
public class AparapiSoftmaxLossFunction implements LossFunction
{
	private static final long serialVersionUID = 1L;

	private NegativeLogProbability nlp;
	private SoftmaxLoss softmaxLoss;

	@Override
	public void getLossFunctionDerivative(Tensor activation, Tensor target, Tensor result)
	{
		if (softmaxLoss == null)
		{
			softmaxLoss = new SoftmaxLoss(activation, target, result);
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(softmaxLoss, activation.getDimensions()[0]);
	}

	@Override
	public float getLossFunction(Tensor activation, Tensor target)
	{
		if (nlp == null)
		{
			nlp = new NegativeLogProbability(activation, target, TensorFactory.tensor(activation.getDimensions()[0]));
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(nlp, activation.getDimensions()[0]);

		float result = 0;
		for (int i = 0; i < nlp.result.length; i++)
		{
			result += nlp.result[i];
		}

		return result;
	}

	public static class SoftmaxLoss extends AparapiLossFunction
	{
		private static final long serialVersionUID = 1L;

		private int reverse;

		public SoftmaxLoss(Tensor activation, Tensor target, Tensor result)
		{
			super(activation, target, result);
			this.reverse = Environment.getInstance().getRuntimeConfiguration().getReverseSoftmaxLoss() ? 1 : -1;
		}

		@Override
		public void run()
		{
			int id = getGlobalId();
			int activationStart = activationStartPosition + activationRowStep * id;
			int targetStart = targetStartPosition + targetRowStep * id;
			int resultStart = resultStartPosition + resultRowStep * id;
			for (int j = 0; j < activationRowStep; j++)
			{
				result[resultStart + j * resultColumnStep] = reverse * (activation[activationStart + j * activationColumnStep] - target[targetStart + j * targetColumnStep]);
			}
		}

		public int getReverse()
		{
			return reverse;
		}
	}

	public static class NegativeLogProbability extends AparapiLossFunction
	{

		private static final long serialVersionUID = 1L;

		public NegativeLogProbability(Tensor activation, Tensor target, Tensor result)
		{
			super(activation, target, result);
		}

		@Override
		public void run()
		{
			int id = getGlobalId();
			int activationStart = activationStartPosition + activationRowStep * id;
			int targetStart = targetStartPosition + targetRowStep * id;
			float value = 0;

			for (int j = 0; j < activationRowStep; j++)
			{
				value += target[targetStart + j * targetColumnStep] * log(activation[activationStart + j * activationColumnStep]);
			}

			result[id] = -value;
		}
	}
}
