package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;

/**
 *
 * base class for training (used for both supervised and unsupervised learning)
 *
 */
public abstract class Trainer {

	protected NeuralNetwork neuralNetwork;
	protected TrainingInputProvider inputProvider;

	public Trainer(NeuralNetwork neuralNetwork, TrainingInputProvider inputProvider) {
		super();
		this.neuralNetwork = neuralNetwork;
		this.inputProvider = inputProvider;
	}

	public void train() {
		for (int index = 0; !stopTraining(index); index++) {
			learnInput(inputProvider.getNextInput(), index);
		}
	}

	public NeuralNetwork getNeuralNetwork() {
		return neuralNetwork;
	}

	public TrainingInputProvider getInputProvider() {
		return inputProvider;
	}

	protected boolean stopTraining(int index) {
		return index >= inputProvider.getInputSize();
	}

	protected abstract void learnInput(TrainingInputData data, int index);
}
