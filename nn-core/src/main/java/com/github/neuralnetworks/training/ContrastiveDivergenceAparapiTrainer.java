package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public class ContrastiveDivergenceAparapiTrainer extends Trainer<RBM> {

    private Matrix posPhaseVisible;
    private Matrix negPhaseVisible;
    private Matrix posPhaseHidden;
    private Matrix negPhaseHidden;
    private float[] weightUpdates;
    private float[] visibleBiasUpdates;
    private float[] hiddenBiasUpdates;
    private Kernel weightUpdatesKernel;
    private Kernel visibleBiasUpdatesKernel;
    private Kernel hiddenBiasUpdatesKernel;
    private LayerCalculator calculator;
    private int miniBatchSize;
    private Map<Layer, Matrix> results;
    private Set<Layer> calculatedLayers;

    public ContrastiveDivergenceAparapiTrainer() {
	super();
    }

    public ContrastiveDivergenceAparapiTrainer(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	if (miniBatchSize != data.getInput().getColumns()) {
	    miniBatchSize = data.getInput().getColumns();
	    init();
	}

	RBM rbm = getNeuralNetwork();

	Matrix input = data.getInput();

	// required for aparapi
	final int miniBatchSize = input.getColumns();
	final float[] posPhaseVisible = this.posPhaseVisible.getElements();
	final float[] negPhaseVisible = this.negPhaseVisible.getElements();
	final float[] posPhaseHidden = this.posPhaseHidden.getElements();
	final float[] negPhaseHidden = this.negPhaseHidden.getElements();
	final float[] weights = rbm.getMainConnections().getConnectionGraph().getElements();
	final float learningRate = (float) properties.get(Constants.LEARNING_RATE);
	final float momentum = (float) properties.get(Constants.MOMENTUM);

	// clamp results to visible layer
	System.arraycopy(input.getElements(), 0, posPhaseVisible, 0, posPhaseVisible.length);
	results.put(rbm.getVisibleLayer(), this.posPhaseVisible);

	// calculate positive phase
	results.put(rbm.getHiddenLayer(), this.posPhaseHidden);
	calculatedLayers.clear();
	calculatedLayers.add(rbm.getVisibleLayer());
	calculator.calculate(calculatedLayers, results, rbm.getHiddenLayer());

	// Gibbs sampling
	int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
	for (int i = 0; i < gibbsSamplingCount; i++) {
	    calculatedLayers.clear();
	    calculatedLayers.add(rbm.getHiddenLayer());
	    results.put(rbm.getVisibleLayer(), this.negPhaseVisible);
	    calculator.calculate(calculatedLayers, results, rbm.getVisibleLayer());

	    calculatedLayers.clear();
	    calculatedLayers.add(rbm.getVisibleLayer());
	    results.put(rbm.getHiddenLayer(), this.negPhaseHidden);
	    calculator.calculate(calculatedLayers, results, rbm.getHiddenLayer());
	}

	// update weights
	if (weightUpdatesKernel == null) {
	    final float[] weightUpdates = this.weightUpdates;
	    final int neuronWeightsColumns = rbm.getMainConnections().getConnectionGraph().getColumns();
	    weightUpdatesKernel = new Kernel() {
		@Override
		public void run() {
		    int id = getGlobalId();
		    int visibleId = (id % neuronWeightsColumns) * miniBatchSize;
		    int hiddenId = (id / neuronWeightsColumns) * miniBatchSize;
		    float weightUpdate = 0;
		    for (int i = 0; i < miniBatchSize; i++) {
			weightUpdate += posPhaseHidden[hiddenId + i] * posPhaseVisible[visibleId + i] - negPhaseHidden[hiddenId + i] * negPhaseVisible[visibleId + i];
		    }

		    weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * weightUpdates[id];
		    weights[id] += weightUpdate;
		    weightUpdates[id] = weightUpdate;
		}
	    };
	}
	weightUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	weightUpdatesKernel.execute(weights.length);

	// update visible bias
	if (rbm.getVisibleBiasConnections() != null) {
	    final float[] visibleBiasWeights = rbm.getVisibleBiasConnections().getConnectionGraph().getElements();
	    final float[] visibleBiasUpdates = this.visibleBiasUpdates;
	    if (visibleBiasUpdatesKernel == null) {
		visibleBiasUpdatesKernel = new Kernel() {
		    @Override
		    public void run() {
			int id = getGlobalId();
			float weightUpdate = 0;
			for (int i = 0; i < miniBatchSize; i++) {
			    weightUpdate += posPhaseVisible[id * miniBatchSize + i] - negPhaseVisible[id * miniBatchSize + i];
			}

			weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * visibleBiasUpdates[id];
			visibleBiasWeights[id] += weightUpdate;
			visibleBiasUpdates[id] = weightUpdate;
		    }
		};
	    }

	    visibleBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    visibleBiasUpdatesKernel.execute(visibleBiasWeights.length);
	}

	// update hidden bias
	if (rbm.getHiddenBiasConnections() != null) {
	    final float[] hiddenBiasWeights = rbm.getHiddenBiasConnections().getConnectionGraph().getElements();
	    final float[] hiddenBiasUpdates = this.hiddenBiasUpdates;

	    if (hiddenBiasUpdatesKernel == null) {
		hiddenBiasUpdatesKernel = new Kernel() {
		    @Override
		    public void run() {
			int id = getGlobalId();
			float weightUpdate = 0;
			for (int i = 0; i < miniBatchSize; i++) {
			    weightUpdate += posPhaseHidden[id * miniBatchSize + i] - negPhaseHidden[id * miniBatchSize + i];
			    hiddenBiasUpdates[id] += posPhaseHidden[id * miniBatchSize + i] - negPhaseHidden[id * miniBatchSize + i];
			}

			weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * hiddenBiasUpdates[id];
			hiddenBiasWeights[id] += weightUpdate;
			hiddenBiasUpdates[id] = weightUpdate;
		    }
		};
	    }

	    hiddenBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    hiddenBiasUpdatesKernel.execute(hiddenBiasWeights.length);
	}
    }

    protected void init() {
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
	calculator = new LayerCalculatorImpl();

	RBM neuralNetwork = getNeuralNetwork();
	posPhaseVisible = new Matrix(neuralNetwork.getVisibleLayer().getNeuronCount(), miniBatchSize);
	negPhaseVisible = new Matrix(neuralNetwork.getVisibleLayer().getNeuronCount(), miniBatchSize);
	posPhaseHidden = new Matrix(neuralNetwork.getHiddenLayer().getNeuronCount(), miniBatchSize);
	negPhaseHidden = new Matrix(neuralNetwork.getHiddenLayer().getNeuronCount(), miniBatchSize);
	weightUpdates = new float[neuralNetwork.getMainConnections().getConnectionGraph().getElements().length];

	if (neuralNetwork.getVisibleBiasConnections() != null) {
	    visibleBiasUpdates = new float[neuralNetwork.getVisibleBiasConnections().getConnectionGraph().getElements().length];
	}

	if (neuralNetwork.getHiddenBiasConnections() != null) {
	    hiddenBiasUpdates = new float[neuralNetwork.getHiddenBiasConnections().getConnectionGraph().getElements().length];
	}
    }
}
