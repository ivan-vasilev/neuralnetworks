package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.events.EpochFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchStartedEvent;
import com.github.neuralnetworks.training.events.NewInputEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Properties;

/**
 * Base trainer for learning one input after another
 *
 * @param <N>
 */
public abstract class OneStepTrainer<N extends NeuralNetwork> extends Trainer<N>
{
	private static final long serialVersionUID = 1L;

	private boolean stopTraining;
	private transient boolean skipCurrentMiniBatch;

	public OneStepTrainer()
	{
		super();
	}

	public OneStepTrainer(Properties properties)
	{
		super(properties);
	}


	@Override
	public void train()
	{
		triggerEvent(new TrainingStartedEvent(this));

		stopTraining = false;
		skipCurrentMiniBatch = false;

		if (getRandomInitializer() != null)
		{
			getRandomInitializer().initialize(getNeuralNetwork());
		}

		getTrainingInputProvider().reset();

		for (int i = 0, batch = 0, epoch = 1; i < getEpochs() * getTrainingInputProvider().getInputSize() && !stopTraining; i += getTrainingBatchSize(), batch++)
		{
			TrainingInputData input = getInput();

			triggerEvent(new NewInputEvent(this, input, getActivations(), batch));

			getTrainingInputProvider().populateNext(input);

			triggerEvent(new MiniBatchStartedEvent(this, input, getActivations(), batch));

			if (!getSkipCurrentMiniBatch())
			{
				learnInput(batch);
			} else
			{
				setSkipCurrentMiniBatch(false);
			}

			triggerEvent(new MiniBatchFinishedEvent(this, input, getActivations(), batch, getTrainingInputProvider()));

			if ((i + getTrainingBatchSize()) / getTrainingInputProvider().getInputSize() >= epoch)
			{
				triggerEvent(new EpochFinishedEvent(this, input, getActivations(), (i + getTrainingBatchSize()) / getTrainingInputProvider().getInputSize(), epoch));
				epoch++;
				getTrainingInputProvider().reset();
			}
		}

		triggerEvent(new TrainingFinishedEvent(this));
	}

	public void stopTraining()
	{
		stopTraining = true;
	}

	@Override
	public void setSkipCurrentMiniBatch(boolean skipCurrentMiniBatch)
	{
		this.skipCurrentMiniBatch = skipCurrentMiniBatch;
	}

	@Override
	public boolean getSkipCurrentMiniBatch()
	{
		return skipCurrentMiniBatch;
	}

	/**
	 * Learning of one batch of examples
	 */
	protected abstract void learnInput(int batch);

	/**
	 * @return the input data to be populated
	 */
	protected abstract TrainingInputData getInput();

	/**
	 * @return activations
	 */
	protected abstract ValuesProvider getActivations();

}
