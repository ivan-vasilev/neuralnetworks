package com.github.neuralnetworks.training.events;

import static org.slf4j.LoggerFactory.getLogger;

import org.slf4j.Logger;

import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.Trainer;

/**
 * @author tmey
 */
public class LinearLearnRateDecrListener implements TrainingEventListener
{
	private static final long serialVersionUID = 3701545154187049754L;

	private static final Logger logger = getLogger(LinearLearnRateDecrListener.class);

	private int changeInterval = 1;
	private float reductionFactor = 10;

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof EpochFinishedEvent)
		{
			int epoch = ((EpochFinishedEvent) event).getEpoch();
			if (epoch % 8 == 0)
			{
				logger.info("Change hyperparameters");
				Trainer<?> t = (Trainer<?>) event.getSource();
				Hyperparameters hp = t.getHyperparameters();

				t.getNeuralNetwork().getConnections().stream().filter(c -> c instanceof WeightsConnections && hp.getLearningRate(c) > 0.00001f)
						.forEach(c -> {
							hp.setLearningRate(c, hp.getLearningRate(c) / reductionFactor);
						});

				hp.setDefaultLearningRate(hp.getDefaultLearningRate() / reductionFactor);
			}
		}
	}

	public int getChangeInterval()
	{
		return changeInterval;
	}

	public float getReductionFactor()
	{
		return reductionFactor;
	}

	public void setChangeInterval(int changeInterval)
	{
		this.changeInterval = changeInterval;
	}

	public void setReductionFactor(float reductionFactor)
	{
		this.reductionFactor = reductionFactor;
	}
}
