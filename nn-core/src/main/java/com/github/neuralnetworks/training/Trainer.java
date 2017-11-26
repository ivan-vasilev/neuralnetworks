package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.calculation.NetworkCalculatorImpl;
import com.github.neuralnetworks.training.events.PhaseFinishedEvent;
import com.github.neuralnetworks.training.events.PhaseStartedEvent;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for training (used for both supervised and unsupervised learning)
 */
public abstract class Trainer<N extends NeuralNetwork> implements NetworkCalculator<N>
{

	private static final long serialVersionUID = 1L;

	/**
	 * Properties for the training (for example learnig rate, weight decay etc)
	 */
	protected Properties properties;

	public Trainer()
	{
		super();
	}

	public Trainer(Properties properties)
	{
		super();
		this.properties = properties;
	}

	/**
	 * Training method
	 */
	public abstract void train();

	/**
	 * Reset 
	 */
	public abstract void reset();

	/**
	 * The network is tested via the testing input provider and the training error is aggregated for each example.
	 */
	public void test()
	{
		NetworkCalculatorImpl<N> nc = new NetworkCalculatorImpl<N>(getProperties())
		{
			private static final long serialVersionUID = 1L;

			@Override
			public PhaseStartedEvent phaseStartedEvent()
			{
				return new TestingStartedEvent(Trainer.this);
			}

			@Override
			public PhaseFinishedEvent phaseFinishedEvent()
			{
				return new TestingFinishedEvent(Trainer.this);
			}
		};

		nc.calculate(getTestingInputProvider());
	}

	@Override
	public Properties getProperties()
	{
		return properties;
	}

	public void setProperties(Properties properties)
	{
		this.properties = properties;
	}

	public TrainingInputProvider getTrainingInputProvider()
	{
		return properties.getParameter(Constants.TRAINING_INPUT_PROVIDER);
	}

	public void setTrainingInputProvider(TrainingInputProvider trainingInputProvider)
	{
		properties.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingInputProvider);
	}

	public TrainingInputProvider getTestingInputProvider()
	{
		return getProperties().getParameter(Constants.TESTING_INPUT_PROVIDER);
	}

	public void setTestingInputProvider(TrainingInputProvider testingInputProvider)
	{
		getProperties().setParameter(Constants.TESTING_INPUT_PROVIDER, testingInputProvider);
	}

	public Hyperparameters getHyperparameters()
	{
		return properties.getParameter(Constants.HYPERPARAMETERS);
	}

	public void setHyperparameters(Hyperparameters hyperparameters)
	{
		properties.setParameter(Constants.HYPERPARAMETERS, hyperparameters);
	}

	public NNRandomInitializer getRandomInitializer()
	{
		return properties.getParameter(Constants.RANDOM_INITIALIZER);
	}

	public void setRandomInitializer(NNRandomInitializer randomInitializer)
	{
		properties.setParameter(Constants.RANDOM_INITIALIZER, randomInitializer);
	}

	public Integer getTrainingBatchSize()
	{
		return properties.getParameter(Constants.TRAINING_BATCH_SIZE);
	}

	public void setTrainingBatchSize(int batchSize)
	{
		properties.setParameter(Constants.TRAINING_BATCH_SIZE, batchSize);
	}

	public Integer getEpochs()
	{
		return properties.getParameter(Constants.EPOCHS) != null ? properties.getParameter(Constants.EPOCHS) : 1;
	}

	public void setEpochs(int epochs)
	{
		properties.setParameter(Constants.EPOCHS, epochs);
	}

	protected boolean stopTraining(int index)
	{
		return index >= getTestingInputProvider().getInputSize();
	}
}
