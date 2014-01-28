package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Properties;

/**
 * Base trainer for learning one input after another
 *
 * @param <N>
 */
public abstract class OneStepTrainer<N extends NeuralNetwork> extends Trainer<N> {

    public OneStepTrainer() {
	super();
    }

    public OneStepTrainer(Properties properties) {
	super(properties);
    }

    @Override
    public void train() {
	triggerEvent(new TrainingStartedEvent(this));

	initializeWithRandom();

	int batch = 0;
	TrainingInputData input = null;
	while ((input = getTrainingInputProvider().getNextInput()) != null) {
	    learnInput(input, batch++);
	    triggerEvent(new MiniBatchFinishedEvent(this, input));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    /**
     * Learning of one batch of examples
     */
    protected abstract void learnInput(TrainingInputData data, int batch);
}
