package com.github.neuralnetworks.training;

import java.util.List;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public abstract class MiniBatchTrainer<N extends NeuralNetwork> extends Trainer<N> {

	private List<TrainingInputData> currentBatch;

	public MiniBatchTrainer() {
		super();
	}

	public MiniBatchTrainer(Properties properties) {
		super(properties);
	}

	@Override
	protected void learnInput(TrainingInputData data, int index) {
		currentBatch.add(data);
		int batchSize = (int) properties.get(Constants.BATCH_SIZE);
		if (index % batchSize == 0 || stopTraining(index)) {
			learnMiniBatchInput(currentBatch, index);
			currentBatch.clear();
		}
	}

	protected abstract void learnMiniBatchInput(List<TrainingInputData> data, int index);
}
