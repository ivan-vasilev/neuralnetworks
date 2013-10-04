package com.github.neuralnetworks.training;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

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
		for (int index = 0; !stopTraining(index); index++) {
			learnInput(getTrainingInputProvider().getNextInput(), index);
		}
	}

	public float test() {
		TrainingInputProvider ip = getTestingInputProvider();
		NeuralNetwork n = getNeuralNetwork();
		LayerCalculator c = getTestLayerCalculator();

		OutputError e = getOutputError();
		e.setTotalNetworkError(0);

		Map<Layer, float[]> calculatedLayers = new HashMap<>();
		TrainingInputData input = null;

		while ((input = ip.getNextInput()) != null) {
			for (float[] r : calculatedLayers.values()) {
				Arrays.fill(r, 0);
			}

			float[] r = calculatedLayers.get(n.getInputLayer());
			if (r == null) {
				r = new float[n.getInputLayer().getNeuronCount()];
				calculatedLayers.put(n.getInputLayer(), r);
			}
			System.arraycopy(input.getInput(), 0, r, 0, r.length);

			c.calculate(calculatedLayers, n.getOutputLayer());
			e.delta(calculatedLayers.get(n.getOutputLayer()), input.getTarget());
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

	public LayerCalculator getTestLayerCalculator() {
		return properties.getParameter(Constants.LAYER_CALCULATOR);
	}

	public void setTestLayerCalculator(LayerCalculator layerCalculator) {
		properties.setParameter(Constants.TEST_PROPAGATION, layerCalculator);
	}

	protected boolean stopTraining(int index) {
		return index >= getTestingInputProvider().getInputSize();
	}

	protected void initializeWithRandom() {
		if (properties.containsKey(Constants.RANDOM_INITIALIZER)) {
			N nn = getNeuralNetwork();
			RandomInitializer r = properties.getParameter(Constants.RANDOM_INITIALIZER);
			for (Connections c : nn.getConnections()) {
				r.initialize(c.getConnectionGraph().getWeights());
			}
		}
	}

	protected abstract void learnInput(TrainingInputData data, int index);
}
