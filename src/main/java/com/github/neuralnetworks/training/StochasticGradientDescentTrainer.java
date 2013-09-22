package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.Propagation;

public class StochasticGradientDescentTrainer extends Trainer {

	protected Propagation inputPropagation;
	protected Propagation errorPropagation;

	public StochasticGradientDescentTrainer(NeuralNetwork neuralNetwork, TrainingInputProvider inputProvider, Propagation inputPropagation, Propagation errorPropagation) {
		super(neuralNetwork, inputProvider);
		this.inputPropagation = inputPropagation;
		this.errorPropagation = errorPropagation;
	}

	@Override
	protected void learnInput(TrainingInputData data, int index) {
		inputPropagation.reset();
		inputPropagation.getCalculated().put(neuralNetwork.getInputLayer(), data.getInput());
		inputPropagation.propagate();
	}
}
