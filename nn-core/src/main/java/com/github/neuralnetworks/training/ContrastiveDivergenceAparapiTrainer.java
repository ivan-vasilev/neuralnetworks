package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionGraphCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class ContrastiveDivergenceAparapiTrainer extends MiniBatchTrainer<RBM> {

	private float[] posPhaseVisible;
	private float[] negPhaseVisible;
	private float[] posPhaseHidden;
	private float[] negPhaseHidden;
	private float[] weightUpdates;
	private float[] visibleBiasUpdates;
	private float[] hiddenBiasUpdates;

	public ContrastiveDivergenceAparapiTrainer() {
		super();
	}

	public ContrastiveDivergenceAparapiTrainer(Properties properties) {
		super(properties);
		init();
	}

	/**
	 * initialize
	 */
	protected void init() {
		RBM neuralNetwork = getNeuralNetwork();
		posPhaseVisible = new float[neuralNetwork.getVisibleLayer().getNeuronCount()];
		negPhaseVisible = new float[neuralNetwork.getVisibleLayer().getNeuronCount()];
		posPhaseHidden = new float[neuralNetwork.getHiddenLayer().getNeuronCount()];
		negPhaseHidden = new float[neuralNetwork.getHiddenLayer().getNeuronCount()];
		weightUpdates = new float[neuralNetwork.getMainConnections().getConnectionGraph().getWeights().length];

		if (neuralNetwork.getVisibleBiasConnections() != null) {
			visibleBiasUpdates = new float[neuralNetwork.getVisibleBiasConnections().getConnectionGraph().getWeights().length];
		}

		if (neuralNetwork.getHiddenBiasConnections() != null) {
			hiddenBiasUpdates = new float[neuralNetwork.getHiddenBiasConnections().getConnectionGraph().getWeights().length];
		}
	}

	@Override
	protected void learnMiniBatchInput(List<TrainingInputData> data, int index) {
		RBM neuralNetwork = getNeuralNetwork();

		// nullify weights
		new Kernel() {
			@Override
			public void run() {
				weightUpdates[getGlobalId()] = 0;
			}
		}.execute(weightUpdates.length);

		if (neuralNetwork.getVisibleBiasConnections() != null) {
			new Kernel() {
				@Override
				public void run() {
					visibleBiasUpdates[getGlobalId()] = 0;
				}
			}.execute(visibleBiasUpdates.length);
		}

		if (neuralNetwork.getHiddenBiasConnections() != null) {
			new Kernel() {
				@Override
				public void run() {
					hiddenBiasUpdates[getGlobalId()] = 0;
				}
			}.execute(hiddenBiasUpdates.length);
		}

		ConnectionGraphCalculator calculator = new ConnectionGraphCalculator(neuralNetwork.getConnections().get(0));
		Map<Layer, float[]> results = new HashMap<Layer, float[]>();
		final int neuronWeightsCount = neuralNetwork.getMainConnections().getConnectionGraph().getNeuronWeightsCount();

		for (TrainingInputData d : data) {
			results.clear();

			// clamp results to visible layer
			System.arraycopy(d.getInput(), 0, posPhaseVisible, 0, posPhaseVisible.length);
			results.put(neuralNetwork.getVisibleLayer(), posPhaseVisible);

			// calculate positive phase
			results.put(neuralNetwork.getHiddenLayer(), posPhaseHidden);
			calculator.calculate(results, neuralNetwork.getHiddenLayer());

			// Gibbs sampling
			int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
			for (int i = 0; i < gibbsSamplingCount; i++) {
				results.put(neuralNetwork.getVisibleLayer(), negPhaseVisible);
				calculator.calculate(results, neuralNetwork.getVisibleLayer());
				results.put(neuralNetwork.getHiddenLayer(), negPhaseHidden);
				calculator.calculate(results, neuralNetwork.getHiddenLayer());
			}

			// add to existing update
			new Kernel() {
				@Override
				public void run() {
					int id = getGlobalId();
					int visibleId = id / neuronWeightsCount;
					int hiddenId = id % neuronWeightsCount;
					weightUpdates[id] = weightUpdates[id] + posPhaseHidden[hiddenId] * posPhaseVisible[visibleId] - negPhaseHidden[hiddenId] * negPhaseVisible[visibleId];
				}
			}.execute(neuralNetwork.getMainConnections().getConnectionGraph().getWeights().length);

			// visible bias
			if (neuralNetwork.getVisibleBiasConnections() != null) {
				new Kernel() {
					@Override
					public void run() {
						int id = getGlobalId();
						visibleBiasUpdates[id] = visibleBiasUpdates[id] + posPhaseVisible[id] - negPhaseVisible[id];
					}
				}.execute(neuralNetwork.getVisibleBiasConnections().getConnectionGraph().getWeights().length);
			}

			// hidden bias
			if (neuralNetwork.getHiddenBiasConnections() != null) {
				new Kernel() {
					@Override
					public void run() {
						int id = getGlobalId();
						hiddenBiasUpdates[id] = hiddenBiasUpdates[id] + posPhaseHidden[id] - negPhaseHidden[id];
					}
				}.execute(neuralNetwork.getHiddenBiasConnections().getConnectionGraph().getWeights().length);
			}
		}

		// update weights
		final int miniBatchSize = data.size();
		final float[] weights = neuralNetwork.getMainConnections().getConnectionGraph().getWeights();
		final float learningRate = (float) properties.get(Constants.LEARNING_RATE);
		new Kernel() {
			@Override
			public void run() {
				int id = getGlobalId();
				weights[id] = weights[id] + learningRate * (weightUpdates[id] / miniBatchSize);
			}
		}.execute(weights.length);

		// update visible bias
		if (neuralNetwork.getVisibleBiasConnections() != null) {
			final float[] visibleBiasWeights = neuralNetwork.getVisibleBiasConnections().getConnectionGraph().getWeights();
			new Kernel() {
				@Override
				public void run() {
					int id = getGlobalId();
					visibleBiasWeights[id] = visibleBiasWeights[id] + learningRate * visibleBiasUpdates[id];
				}
			}.execute(neuralNetwork.getVisibleBiasConnections().getConnectionGraph().getWeights().length);
		}

		// update hidden bias
		if (neuralNetwork.getHiddenBiasConnections() != null) {
			final float[] hiddenBiasWeights = neuralNetwork.getHiddenBiasConnections().getConnectionGraph().getWeights();
			new Kernel() {
				@Override
				public void run() {
					int id = getGlobalId();
					hiddenBiasWeights[id] = hiddenBiasWeights[id] + learningRate * hiddenBiasUpdates[id];
				}
			}.execute(neuralNetwork.getHiddenBiasConnections().getConnectionGraph().getWeights().length);
		}
	}
}
