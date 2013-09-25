package com.github.neuralnetworks.training;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Constants;

public abstract class MiniBatchTrainer extends Trainer {

	private List<TrainingInputData> currentBatch;
	private int batchSize;

	public MiniBatchTrainer(NeuralNetwork neuralNetwork, TrainingInputProvider inputProvider, Map<String, Object> properties) {
		super(neuralNetwork, inputProvider, properties);
		this.batchSize = (int) properties.get(Constants.BATCH_SIZE);
	}

	@Override
	protected void learnInput(TrainingInputData data, int index) {
		currentBatch.add(data);
		if (index % batchSize == 0 || stopTraining(index)) {
			learnMiniBatchInput(currentBatch, index);
			currentBatch.clear();
		}
	}

	protected abstract void learnMiniBatchInput(List<TrainingInputData> data, int index);
}
