package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.Propagation;
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
		Propagation p = getTestPropagation();

		OutputError e = getOutputError();
		e.setTotalNetworkError(0);

		TrainingInputData input = null;
		while ((input = ip.getNextInput()) != null) {
			p.propagate(input.getInput(), n.getInputLayer());
			float[] networkOutput = p.getCalculated().get(n.getOutputLayer());
			e.delta(networkOutput, input.getTarget());
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

	public Propagation getTestPropagation() {
		return properties.getParameter(Constants.TEST_PROPAGATION);
	}

	public void setTestPropagation(Propagation testPropagation) {
		properties.setParameter(Constants.TEST_PROPAGATION, testPropagation);
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
