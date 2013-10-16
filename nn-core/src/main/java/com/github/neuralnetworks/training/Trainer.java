package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * 
 * base class for training (used for both supervised and unsupervised learning)
 * 
 */
public abstract class Trainer<N extends NeuralNetwork> {

    protected Properties properties;

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
	}
    }

    public float test() {
	TrainingInputProvider ip = getTestingInputProvider();
	NeuralNetwork n = getNeuralNetwork();
	LayerCalculator c = getLayerCalculator();

	OutputError e = getOutputError();

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(n.getInputLayer());

	Map<Layer, Matrix> results = new HashMap<>();
	TrainingInputData input = null;

	while ((input = ip.getNextInput()) != null) {
	    for (Matrix m : results.values()) {
		Util.fillArray(m.getElements(), 0);
	    }

	    results.put(n.getInputLayer(), input.getInput());

	    c.calculate(calculatedLayers, results, n.getOutputLayer());
	    e.addItem(results.get(n.getOutputLayer()), input.getTarget());
	}

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

    protected boolean stopTraining(int index) {
	return index >= getTestingInputProvider().getInputSize();
    }

    public int getMiniBatchSize() {
	return properties.getParameter(Constants.MINI_BATCH_SIZE);
    }

    public void setMiniBatchSize(int miniBatchSize) {
	properties.setParameter(Constants.MINI_BATCH_SIZE, miniBatchSize);
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
}
