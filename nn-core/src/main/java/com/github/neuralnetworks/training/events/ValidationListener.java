package com.github.neuralnetworks.training.events;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.NetworkCalculatorImpl;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * Listener for early stopping of the training
 */
public class ValidationListener implements TrainingEventListener
{

	private static final long serialVersionUID = 1L;

	private static final Logger logger = LoggerFactory.getLogger(ValidationListener.class);

	/**
	 * input provider for the cross-validation data
	 */
	private TrainingInputProvider inputProvider;

	/**
	 * What error is considered acceptable to stop
	 */
	private float acceptanceError;
	protected float prevError;
	private int validationEpochs;

	private boolean isTraining;

	/**
	 * @param inputProvider
	 *          - validation data
	 * @param acceptanceError
	 *          - if validation error is below threshold then stop
	 */
	public ValidationListener(TrainingInputProvider inputProvider, float acceptanceError)
	{
		super();
		this.inputProvider = inputProvider;
		this.acceptanceError = acceptanceError;
		this.validationEpochs = 1;
	}

	/**
	 */
	/**
	 * @param inputProvider
	 *          - validation data
	 * @param acceptanceError
	 *          - if validation error is below threshold then stop
	 * @param validationEpochs
	 *          - perform validation each validationEpochs epochs
	 */
	public ValidationListener(TrainingInputProvider inputProvider, float acceptanceError, int validationEpochs)
	{
		super();
		this.inputProvider = inputProvider;
		this.acceptanceError = acceptanceError;
		this.validationEpochs = validationEpochs;
	}

	public ValidationListener(TrainingInputProvider inputProvider)
	{
		super();
		this.inputProvider = inputProvider;
		this.acceptanceError = 0;
		this.validationEpochs = 1;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event.getSource() != this)
		{
			if (event instanceof TrainingStartedEvent)
			{
				isTraining = true;
			} else if (event instanceof TrainingFinishedEvent)
			{
				isTraining = false;
			} else if (event instanceof EpochFinishedEvent && isTraining && event.getSource() instanceof OneStepTrainer && ((EpochFinishedEvent) event).getEpoch() % validationEpochs == 0)
			{
				OneStepTrainer<?> t = (OneStepTrainer<?>) event.getSource();
				if (t.getOutputError() != null)
				{
					NetworkCalculatorImpl<?> nc = new NetworkCalculatorImpl<NeuralNetwork>(t.getProperties())
					{
						private static final long serialVersionUID = 1L;

						@Override
						public PhaseStartedEvent phaseStartedEvent()
						{
							return new ValidationStartedEvent(this);
						}

						@Override
						public PhaseFinishedEvent phaseFinishedEvent()
						{
							return new ValidationFinishedEvent(this);
						}
					};

					nc.triggerEvent(new TrainingFinishedEvent(t));
					nc.calculate(inputProvider);
					if (acceptanceError > 0 && nc.getOutputError().getTotalNetworkError() <= acceptanceError)
					{
						t.stopTraining();
					}

					if (prevError > 0 && prevError < nc.getOutputError().getTotalNetworkError())
					{
						logger.warn("WARNING: previous error < currentError " + prevError + " < " + nc.getOutputError().getTotalNetworkError());
						errorIncrease((EpochFinishedEvent) event, prevError, nc.getOutputError().getTotalNetworkError());
					}

					prevError = nc.getOutputError().getTotalNetworkError();

					nc.triggerEvent(new TrainingStartedEvent(t));
				}
			}
		}
	}

	@SuppressWarnings("unused")
	protected void errorIncrease(EpochFinishedEvent event, float prevError, float currentError)
	{
	}
}
