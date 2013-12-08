package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
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
	super.train();

	TrainingInputData input = null;
	while ((input = getTrainingInputProvider().getNextInput()) != null) {
	    learnInput(input);
	    triggerEvent(new SampleFinishedEvent(this, input));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    /**
     * Learning of one batch of examples
     */
    protected abstract void learnInput(TrainingInputData data);
}
