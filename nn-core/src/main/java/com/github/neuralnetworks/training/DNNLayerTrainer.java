package com.github.neuralnetworks.training;

import java.util.Map;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for deep network trainer
 */
public class DNNLayerTrainer extends Trainer<DNN<? extends NeuralNetwork>>
{

	private static final long serialVersionUID = 1L;

	public DNNLayerTrainer(Properties properties)
	{
		super(properties);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.github.neuralnetworks.training.Trainer#train()
	 * Child netwokrs are trained in sequential order. Each network has it's own Trainer.
	 */
	@Override
	public void train()
	{
		triggerEvent(new TrainingStartedEvent(this));

		DNN<?> dnn = getNeuralNetwork();

		for (NeuralNetwork nn : dnn.getNeuralNetworks())
		{
			DeepTrainerTrainingInputProvider inputProvider = new DeepTrainerTrainingInputProvider(getTrainingInputProvider(), dnn, nn, getTrainingBatchSize());
			inputProvider.reset();
			OneStepTrainer<?> trainer = getTrainers().get(nn);
			trainer.setTrainingInputProvider(inputProvider);
			trainer.train();

			triggerEvent(new LayerTrainingFinished(this, trainer));
		}

		triggerEvent(new TrainingFinishedEvent(this));
	}

	public Map<NeuralNetwork, OneStepTrainer<?>> getTrainers()
	{
		return properties.getParameter(Constants.LAYER_TRAINERS);
	}

	@Override
	public Integer getTrainingBatchSize()
	{
		return getTrainers().values().iterator().next().getTrainingBatchSize();
	}

	@Override
	public Integer getTestBatchSize()
	{
		return getTrainers().values().iterator().next().getTestBatchSize();
	}

	@Override
	public Integer getEpochs()
	{
		return getTrainers().values().iterator().next().getEpochs();
	}

	@Override
	public void reset()
	{
		if (getTrainingInputProvider() != null)
		{
			getTrainingInputProvider().reset();
		}

		getNeuralNetwork().getNeuralNetworks().forEach(nn -> getTrainers().get(nn).reset());
	}

	@Override
	public void setSkipCurrentMiniBatch(boolean skipCurrentMiniBatch)
	{
		getTrainers().values().forEach(t -> t.setSkipCurrentMiniBatch(skipCurrentMiniBatch));
	}

	@Override
	public boolean getSkipCurrentMiniBatch()
	{
		return false;
	}

	/**
	 * Triggered when a nested neural network has finished training
	 */
	public static class LayerTrainingFinished extends TrainingEvent
	{

		private static final long serialVersionUID = 2155527437110587968L;

		public OneStepTrainer<?> currentTrainer;

		public LayerTrainingFinished(Trainer<?> source, OneStepTrainer<?> currentTrainer)
		{
			super(source);
			this.currentTrainer = currentTrainer;
		}
	}
}
