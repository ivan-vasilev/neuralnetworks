package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.random.RandomInitializer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Base class for training (used for both supervised and unsupervised learning)
 */
public abstract class Trainer<N extends NeuralNetwork> {

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
	    Map<Layer, Matrix> results = new HashMap<>();
	    TrainingInputData input = null;
	    
	    while ((input = ip.getNextInput()) != null) {
		calculatedLayers.clear();
		calculatedLayers.add(n.getInputLayer());
		results.put(n.getInputLayer(), input.getInput());
		n.getLayerCalculator().calculate(n, n.getOutputLayer(), calculatedLayers, results);

		if (getOutputError() != null) {
		    getOutputError().addItem(results.get(n.getOutputLayer()), input.getTarget());
		}

		triggerEvent(new MiniBatchFinishedEvent(this, input));
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
    
    public RandomInitializer getRandomInitializer() {
	return properties.getParameter(Constants.RANDOM_INITIALIZER);
    }
    
    public void setRandomInitializer(RandomInitializer randomInitializer) {
	properties.setParameter(Constants.RANDOM_INITIALIZER, randomInitializer);
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
	    for (TrainingEventListener l : listeners) {
		l.handleEvent(event);
	    }
	}
    }

    protected boolean stopTraining(int index) {
	return index >= getTestingInputProvider().getInputSize();
    }

    /**
     * Random initialization of weights
     */
    protected void initializeWithRandom() {
	if (getRandomInitializer() != null) {
	    N nn = getNeuralNetwork();
	    RandomInitializer r = getRandomInitializer();
	    for (Connections c : nn.getConnections()) {
		if (c instanceof GraphConnections) {
		    r.initialize(((GraphConnections) c).getConnectionGraph().getElements());
		}
	    }
	}
    }
}
