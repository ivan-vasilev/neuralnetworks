package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Constants;
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
	    getRandomInitializer().initialize(getNeuralNetwork());;
	}

	getTrainingInputProvider().reset();

	int batch = 0;
	TrainingInputData input = null;
	while ((input = getTrainingInputProvider().getNextInput()) != null && !stopTraining) {
	    learnInput(input, batch++);
	    triggerEvent(new MiniBatchFinishedEvent(this, input, null, batch));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    public void stopTraining() {
	stopTraining = true;
    }

    /**
     * Learning of one batch of examples
     */
    protected abstract void learnInput(TrainingInputData data, int batch);

    public Integer getBatchSize() {
	return properties.getParameter(Constants.BATCH_SIZE);
    }

    public void setBatchSize(int batchSize) {
	properties.setParameter(Constants.BATCH_SIZE, batchSize);
    }
}
