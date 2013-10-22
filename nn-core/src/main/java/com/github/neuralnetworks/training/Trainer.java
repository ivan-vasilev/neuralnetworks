package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * 
 * base class for training (used for both supervised and unsupervised learning)
 * 
 */
public abstract class Trainer<N extends NeuralNetwork> {

    protected Properties properties;
    protected List<TrainingEventListener> listeners;

    public Trainer() {
	super();
    }

    public Trainer(Properties properties) {
	super();
	this.properties = properties;

	// random initialization
	initializeWithRandom();
    }

    public void train() {
	TrainingInputData input = null;
	while ((input = getTrainingInputProvider().getNextInput()) != null) {
	    learnInput(input);
	    triggerEvent(new SampleFinishedEvent(this, input));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    public float test() {
	TrainingInputProvider ip = getTestingInputProvider();
	NeuralNetwork n = getNeuralNetwork();
	LayerCalculator c = getLayerCalculator();

	OutputError e = getOutputError();

	Set<Layer> calculatedLayers = new UniqueList<>();
	Map<Layer, Matrix> results = new HashMap<>();
	TrainingInputData input = null;

	while ((input = ip.getNextInput()) != null) {
	    calculatedLayers.clear();
	    calculatedLayers.add(n.getInputLayer());
	    results.put(n.getInputLayer(), input.getInput());
	    c.calculate(calculatedLayers, results, n.getOutputLayer());
	    e.addItem(results.get(n.getOutputLayer()), input.getTarget());

	    for (Matrix m : results.values()) {
		Util.fillArray(m.getElements(), 0);
	    }

	    triggerEvent(new SampleFinishedEvent(this, input));
	}

	triggerEvent(new TestingFinishedEvent(this));

	return e.getTotalNetworkError();
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

    public LayerCalculator getLayerCalculator() {
	return properties.getParameter(Constants.LAYER_CALCULATOR);
    }

    public void setLayerCalculator(LayerCalculator layerCalculator) {
	properties.setParameter(Constants.LAYER_CALCULATOR, layerCalculator);
    }

    public void addEventListener(TrainingEventListener listener) {
	if (listeners == null) {
	    listeners = new ArrayList<>();
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
	    for (TrainingEventListener l : listeners) {
		l.handleEvent(event);
	    }
	}
    }

    protected boolean stopTraining(int index) {
	return index >= getTestingInputProvider().getInputSize();
    }

    protected void initializeWithRandom() {
	if (properties.containsKey(Constants.RANDOM_INITIALIZER)) {
	    N nn = getNeuralNetwork();
	    RandomInitializer r = properties.getParameter(Constants.RANDOM_INITIALIZER);
	    for (Connections c : nn.getConnections()) {
		r.initialize(c.getConnectionGraph().getElements());
	    }
	}
    }

    protected abstract void learnInput(TrainingInputData data);

    public static class TrainingFinishedEvent extends TrainingEvent {

	private static final long serialVersionUID = -5239379347414855784L;

	public TrainingFinishedEvent(Trainer<?> source) {
	    super(source);
	}
    }

    public static class SampleFinishedEvent extends TrainingEvent {

	private static final long serialVersionUID = -5239379347414855784L;

	public TrainingInputData data;

	public SampleFinishedEvent(Trainer<?> source, TrainingInputData data) {
	    super(source);
	    this.data = data;
	}
    }

    public static class TestingFinishedEvent extends TrainingEvent {

	private static final long serialVersionUID = -5239379347414855784L;

	public TestingFinishedEvent(Trainer<?> source) {
	    super(source);
	}
    }
}
