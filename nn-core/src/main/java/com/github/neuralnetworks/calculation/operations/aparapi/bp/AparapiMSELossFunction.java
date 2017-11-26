package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.util.Environment;

/**
 * Mean squared error derivative
 */
public class AparapiMSELossFunction implements LossFunction
{
	private static final long serialVersionUID = 1L;

	private MSE mse;
	private MSEDerivative mseDerivative;

	@Override
	public void getLossFunctionDerivative(Tensor activation, Tensor target, Tensor result)
	{
		if (mseDerivative == null)
		{
			mseDerivative = new MSEDerivative(activation, target, result);
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(mseDerivative, activation.getDimensions()[0]);
	}

	@Override
	public float getLossFunction(Tensor activation, Tensor target)
	{
		if (mse == null)
		{
			mse = new MSE(activation, target, TensorFactory.tensor(activation.getDimensions()[0]));
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(mse, activation.getDimensions()[0]);

		float result = 0;
		for (int i = 0; i < mse.result.length; i++)
		{
			result += mse.result[i];
		}

		return result;
	}

	public static class MSEDerivative extends AparapiLossFunction
	{

		private static final long serialVersionUID = 1L;

		public MSEDerivative(Tensor activation, Tensor target, Tensor result)
		{
			super(activation, target, result);
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
				result[resultStart + j * resultColumnStep] = target[targetStart + j * targetColumnStep] - activation[activationStart + j * activationColumnStep];
			}
		}
	}

	public static class MSE extends AparapiLossFunction
	{

		private static final long serialVersionUID = 1L;

		public MSE(Tensor activation, Tensor target, Tensor result)
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
				float diff = target[targetStart + j * targetColumnStep] - activation[activationStart + j * activationColumnStep];
				value += 0.5 * diff * diff;
			}

			result[resultStartPosition + id] = -value;
		}
	}
}
