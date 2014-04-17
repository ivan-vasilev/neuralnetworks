package com.github.neuralnetworks.training;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Base class for training (used for both supervised and unsupervised learning)
 */
public abstract class Trainer<N extends NeuralNetwork> implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Properties for the training (for example learnig rate, weight decay etc)
     */
    protected Properties properties;

    /**
     * Training event listeners
     */
    protected List<TrainingEventListener> listeners;

    public Trainer() {
	super();
    }

    public Trainer(Properties properties) {
	super();
	this.properties = properties;
    }

    /**
     * Training method
     */
    public abstract void train();

    /**
     * The network is tested via the testing input provider and the training error is aggregated for each example.
     */
    public void test() {
	TrainingInputProvider ip = getTestingInputProvider();
	NeuralNetwork n = getNeuralNetwork();

	if (ip != null && n != null && n.getLayerCalculator() != null) {
	    ip.reset();

	    triggerEvent(new TestingStartedEvent(this));

	    Set<Layer> calculatedLayers = new UniqueList<>();
	    ValuesProvider results = TensorFactory.tensorProvider(n, getTestBatchSize(), Environment.getInstance().getUseDataSharedMemory());

	    OutputError oe = getOutputError();
	    if (oe != null) {
		oe.reset();
		results.add(oe, results.get(n.getOutputLayer()).getDimensions());
	    }

	    TrainingInputData input = new TrainingInputDataImpl(results.get(n.getInputLayer()), results.get(oe));
	    for (int i = 0; i < ip.getInputSize(); i += getTestBatchSize()) {
		ip.populateNext(input);
		calculatedLayers.clear();
		calculatedLayers.add(n.getInputLayer());
		n.getLayerCalculator().calculate(n, n.getOutputLayer(), calculatedLayers, results);
		
		if (oe != null) {
		    oe.addItem(results.get(n.getOutputLayer()), input.getTarget());
		}
		
		triggerEvent(new MiniBatchFinishedEvent(this, input, results, null));
	    }
	    
	    triggerEvent(new TestingFinishedEvent(this));
	}
    }

    public Properties getProperties() {
	return properties;
    }

    public void setProperties(Properties properties) {
	this.properties = properties;
    }

    public N getNeuralNetwork() {
	return properties.getParameter(Constants.NEURAL_NETWORK);
    }

    public void setNeuralNetwork(N neuralNetwork) {
	properties.setParameter(Constants.NEURAL_NETWORK, neuralNetwork);
    }

    public TrainingInputProvider getTrainingInputProvider() {
	return properties.getParameter(Constants.TRAINING_INPUT_PROVIDER);
    }

    public void setTrainingInputProvider(TrainingInputProvider trainingInputProvider) {
	properties.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingInputProvider);
    }

    public TrainingInputProvider getTestingInputProvider() {
	return properties.getParameter(Constants.TESTING_INPUT_PROVIDER);
    }

    public void setTestingInputProvider(TrainingInputProvider testingInputProvider) {
	properties.setParameter(Constants.TESTING_INPUT_PROVIDER, testingInputProvider);
    }

    public OutputError getOutputError() {
	return properties.getParameter(Constants.OUTPUT_ERROR);
    }

    public void setOutputError(OutputError outputError) {
	properties.setParameter(Constants.OUTPUT_ERROR, outputError);
    }
    
    public NNRandomInitializer getRandomInitializer() {
	return properties.getParameter(Constants.RANDOM_INITIALIZER);
    }
    
    public void setRandomInitializer(NNRandomInitializer randomInitializer) {
	properties.setParameter(Constants.RANDOM_INITIALIZER, randomInitializer);
    }

    public Integer getTrainingBatchSize() {
	return properties.getParameter(Constants.TRAINING_BATCH_SIZE);
    }

    public void setTrainingBatchSize(int batchSize) {
	properties.setParameter(Constants.TRAINING_BATCH_SIZE, batchSize);
    }
    
    public Integer getTestBatchSize() {
	return properties.getParameter(Constants.TEST_BATCH_SIZE) != null ? properties.getParameter(Constants.TEST_BATCH_SIZE) : 1;
    }
    
    public void setTestBatchSize(int batchSize) {
	properties.setParameter(Constants.TEST_BATCH_SIZE, batchSize);
    }
    
    public Integer getEpochs() {
	return properties.getParameter(Constants.EPOCHS) != null ? properties.getParameter(Constants.EPOCHS) : 1;
    }
    
    public void setEpochs(int epochs) {
	properties.setParameter(Constants.EPOCHS, epochs);
    }

    public void addEventListener(TrainingEventListener listener) {
	if (listeners == null) {
	    listeners = new UniqueList<>();
	}

	listeners.add(listener);
    }

    public void removeEventListener(TrainingEventListener listener) {
	if (listeners != null) {
	    listeners.remove(listener);
	}
    }

    protected void triggerEvent(TrainingEvent event) {
	if (listeners != null) {
	    List<TrainingEventListener> listeners = new ArrayList<>(this.listeners);
	    listeners.forEach(l -> l.handleEvent(event));
	}
    }

    protected boolean stopTraining(int index) {
	return index >= getTestingInputProvider().getInputSize();
    }
}
