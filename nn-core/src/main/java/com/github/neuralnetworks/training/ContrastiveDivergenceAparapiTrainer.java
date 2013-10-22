package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

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
    private int miniBatchSize;

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
	Set<Layer> calculatedLayers = new UniqueList<Layer>();

	// nullify weights
	Util.fillArray(weightUpdates, 0);

	if (rbm.getVisibleBiasConnections() != null) {
	    Util.fillArray(visibleBiasUpdates, 0);
	}

	if (rbm.getHiddenBiasConnections() != null) {
	    Util.fillArray(hiddenBiasUpdates, 0);
	}

	LayerCalculatorImpl calculator = new LayerCalculatorImpl();

	// TODO member
	Map<Layer, Matrix> results = new HashMap<>();

	results.clear();

	// clamp results to visible layer
	System.arraycopy(input.getElements(), 0, posPhaseVisible, 0, posPhaseVisible.length);
	results.put(rbm.getVisibleLayer(), this.posPhaseVisible);

	// calculate positive phase
	results.put(rbm.getHiddenLayer(), this.posPhaseHidden);
	calculatedLayers.add(rbm.getVisibleLayer());
	calculator.calculate(calculatedLayers, results, rbm.getHiddenLayer());

	// Gibbs sampling
	int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
	for (int i = 0; i < gibbsSamplingCount; i++) {
	    results.put(rbm.getVisibleLayer(), this.negPhaseVisible);
	    calculatedLayers.clear();
	    calculatedLayers.add(rbm.getHiddenLayer());
	    calculator.calculate(calculatedLayers, results, rbm.getVisibleLayer());
	    results.put(rbm.getHiddenLayer(), this.negPhaseHidden);
	    calculatedLayers.add(rbm.getVisibleLayer());
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
		    for (int i = 0; i < miniBatchSize; i++) {
			weightUpdates[id] += posPhaseHidden[hiddenId + i] * posPhaseVisible[visibleId + i] - negPhaseHidden[hiddenId + i] * negPhaseVisible[visibleId + i];
		    }

		    weights[id] += learningRate * (weightUpdates[id] / miniBatchSize);
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
			for (int i = 0; i < miniBatchSize; i++) {
			    visibleBiasUpdates[id] += posPhaseVisible[id * miniBatchSize + i] - negPhaseVisible[id * miniBatchSize + i];
			}

			visibleBiasWeights[id] += learningRate * (visibleBiasUpdates[id] / miniBatchSize);
		    }
		};
	    }

	    visibleBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    visibleBiasUpdatesKernel.execute(visibleBiasWeights.length);
	}

	// update hidden bias
	if (rbm.getHiddenBiasConnections() != null) {
	    final float[] hiddenBiasWeights = rbm.getHiddenBiasConnections().getConnectionGraph().getElements();
	    final float[] hiddenBiasUpdates = this.visibleBiasUpdates;

	    if (hiddenBiasUpdatesKernel == null) {
		hiddenBiasUpdatesKernel = new Kernel() {
		    @Override
		    public void run() {
			int id = getGlobalId();
			for (int i = 0; i < miniBatchSize; i++) {
			    hiddenBiasUpdates[id] += posPhaseHidden[id * miniBatchSize + i] - negPhaseHidden[id * miniBatchSize + i];
			}

			hiddenBiasWeights[id] += learningRate * (hiddenBiasUpdates[id] / miniBatchSize);
		    }
		};
	    }

	    hiddenBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    hiddenBiasUpdatesKernel.execute(hiddenBiasWeights.length);
	}
    }

    protected void init() {
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
