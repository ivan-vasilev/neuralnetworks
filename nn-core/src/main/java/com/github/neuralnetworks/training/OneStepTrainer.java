package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.training.events.EpochFinishedEvent;
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

    private static final long serialVersionUID = 1L;

    private boolean stopTraining;

    public OneStepTrainer() {
	super();
    }

    public OneStepTrainer(Properties properties) {
	super(properties);
    }

    @Override
    public void train() {
	triggerEvent(new TrainingStartedEvent(this));

	stopTraining = false;

	if (getRandomInitializer() != null) {
	    getRandomInitializer().initialize(getNeuralNetwork());
	}

	getTrainingInputProvider().reset();

	for (int i = 0, batch = 0; i < getEpochs() * getTrainingInputProvider().getInputSize() && !stopTraining; i += getTrainingBatchSize(), batch++) {
	    TrainingInputData input = getInput();
	    getTrainingInputProvider().populateNext(input);
	    learnInput(batch);
	    triggerEvent(new MiniBatchFinishedEvent(this, input, null, batch));

	    if (i % getTrainingInputProvider().getInputSize() == 0) {
		triggerEvent(new EpochFinishedEvent(this, input, null, i / getTrainingInputProvider().getInputSize()));
	    }
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    public void stopTraining() {
	stopTraining = true;
    }

    /**
     * Learning of one batch of examples
     */
    protected abstract void learnInput(int batch);

    /**
     * @return the input data to be populated
     */
    protected abstract TrainingInputData getInput();
}
