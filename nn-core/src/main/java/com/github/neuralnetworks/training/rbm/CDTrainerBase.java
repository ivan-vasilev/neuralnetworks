package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for Contrastive Divergence
 * requires RBMLayerCalculator as the layer calculator. This allows for different implementations of the layer calculator, like GPU/CPU for example
 */
public abstract class CDTrainerBase extends OneStepTrainer<RBM>
{

	private static final long serialVersionUID = 1L;

	private TrainingInputData input;

	public CDTrainerBase(Properties properties)
	{
		super(properties);
	}

	@Override
	protected TrainingInputData getInput()
	{
		if (input == null)
		{
			input = new TrainingInputDataImpl(getLayerCalculator().getPositivePhaseVisible());
		}

		return input;
	}

	@Override
	protected void learnInput(int batch)
	{
		RBM nn = getNeuralNetwork();

		getLayerCalculator().gibbsSampling(nn, getGibbsSamplingCount(), batch == 0 ? true : !getIsPersistent());

		// update weights
		updateWeights();
	}

	public RBMLayerCalculator getLayerCalculator()
	{
		return properties.getParameter(Constants.LAYER_CALCULATOR);
	}

	public void setLayerCalculator(RBMLayerCalculator layerCalculator)
	{
		properties.setParameter(Constants.LAYER_CALCULATOR, layerCalculator);
	}

	public Boolean getIsPersistent()
	{
		return properties.getParameter(Constants.PERSISTENT_CD);
	}

	public void setIsPersistent(boolean isPersistent)
	{
		properties.setParameter(Constants.PERSISTENT_CD, isPersistent);
	}

	public int getGibbsSamplingCount()
	{
		Hyperparameters hp = (Hyperparameters) properties.get(Constants.HYPERPARAMETERS);
		return hp.getDefault(Constants.GIBBS_SAMPLING_COUNT) != null ? (int) hp.getDefault(Constants.GIBBS_SAMPLING_COUNT) : 1;
	}

	@Override
	protected ValuesProvider getActivations()
	{
		return null;
	}

	@Override
	public void reset()
	{
		if (getTrainingInputProvider() != null)
		{
			getTrainingInputProvider().reset();
		}
	}

	protected abstract void updateWeights();
}
