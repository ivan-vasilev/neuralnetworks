package com.github.neuralnetworks.training;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiBinaryRandomSigmoidConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoidConnectionCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public class ContrastiveDivergenceAparapiTrainer extends Trainer<RBM> {

    private Matrix posPhaseVisible;
    private Matrix negPhaseVisible;
    private Matrix posPhaseHidden;
    private Matrix negPhaseHidden;
    private Kernel weightUpdatesKernel;
    private Kernel visibleBiasUpdatesKernel;
    private Kernel hiddenBiasUpdatesKernel;
    private int miniBatchSize;
    private RBMLayerCalculator calculator;
    private ConnectionCalculator hiddenConnectionCalculator;
    private ConnectionCalculator visibleConnectionCalculator;

    public ContrastiveDivergenceAparapiTrainer() {
	super();
    }

    public ContrastiveDivergenceAparapiTrainer(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	posPhaseVisible = data.getInput();

	if (miniBatchSize != data.getInput().getColumns()) {
	    miniBatchSize = data.getInput().getColumns();
	    init();
	}

	RBM rbm = getNeuralNetwork();

	calculator.calculateHiddenLayer(posPhaseVisible, posPhaseHidden, hiddenConnectionCalculator);

	// Gibbs sampling
	int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
	for (int i = 0; i < gibbsSamplingCount; i++) {
	    calculator.calculateVisibleLayer(negPhaseVisible, posPhaseHidden, visibleConnectionCalculator);
	    calculator.calculateHiddenLayer(negPhaseVisible, negPhaseHidden, hiddenConnectionCalculator);
	}

	// update weights
	weightUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	weightUpdatesKernel.execute(rbm.getMainConnections().getConnectionGraph().getElements().length);

	// update visible bias
	if (visibleBiasUpdatesKernel != null) {
	    visibleBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    visibleBiasUpdatesKernel.execute(rbm.getVisibleBiasConnections().getConnectionGraph().getElements().length);
	}

	// update hidden bias
	if (hiddenBiasUpdatesKernel != null) {
	    hiddenBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    hiddenBiasUpdatesKernel.execute(rbm.getHiddenBiasConnections().getConnectionGraph().getElements().length);
	}
    }

    protected void init() {
	hiddenConnectionCalculator = new AparapiBinaryRandomSigmoidConnectionCalculator();
	visibleConnectionCalculator = new AparapiSigmoidConnectionCalculator();

	RBM nn = getNeuralNetwork();
	calculator = new RBMLayerCalculator(nn);

	//this.posPhaseVisible = new Matrix(new float[nn.getVisibleLayer().getNeuronCount()], miniBatchSize);
	this.negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	this.posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	this.negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	float learningRate = properties.getParameter(Constants.LEARNING_RATE);
	float momentum = properties.getParameter(Constants.MOMENTUM);

	Matrix weights = nn.getMainConnections().getConnectionGraph();

	weightUpdatesKernel = new WeightUpdatesKernel(posPhaseVisible.getElements(), posPhaseHidden.getElements(), negPhaseVisible.getElements(), negPhaseHidden.getElements(), weights.getElements(), weights.getColumns(), learningRate, momentum, miniBatchSize);

	if (nn.getVisibleBiasConnections() != null) {
	    visibleBiasUpdatesKernel = new BiasUpdatesKernel(nn.getVisibleBiasConnections().getConnectionGraph().getElements(), posPhaseVisible.getElements(), negPhaseVisible.getElements(), learningRate, momentum, miniBatchSize);
	}

	if (nn.getHiddenBiasConnections() != null) {
	    hiddenBiasUpdatesKernel = new BiasUpdatesKernel(nn.getHiddenBiasConnections().getConnectionGraph().getElements(), posPhaseHidden.getElements(), negPhaseHidden.getElements(), learningRate, momentum, miniBatchSize);
	}
    }

    private static class WeightUpdatesKernel extends Kernel {

	private float[] posPhaseHidden;
	private float[] posPhaseVisible;
	private float[] negPhaseHidden;
	private float[] negPhaseVisible;
	private float[] weights;
	private float[] weightUpdates;
	private int weightColumns;
	private float learningRate;
	private float momentum;
	private int miniBatchSize;

	public WeightUpdatesKernel(float[] posPhaseVisible, float[] posPhaseHidden, float[] negPhaseVisible, float[] negPhaseHidden, float[] weights, int weightColumns, float learningRate, float momentum, int miniBatchSize) {
	    super();
	    this.posPhaseVisible = posPhaseVisible;
	    this.posPhaseHidden = posPhaseHidden;
	    this.negPhaseVisible = negPhaseVisible;
	    this.negPhaseHidden = negPhaseHidden;
	    this.weights = weights;
	    this.weightUpdates = new float[weights.length];
	    this.weightColumns = weightColumns;
	    this.learningRate = learningRate;
	    this.momentum = momentum;
	    this.miniBatchSize = miniBatchSize;
	}

	@Override
	public void run() {
	    int id = getGlobalId();
	    int visibleId = (id % weightColumns) * miniBatchSize;
	    int hiddenId = (id / weightColumns) * miniBatchSize;
	    float weightUpdate = 0;
	    for (int i = 0; i < miniBatchSize; i++) {
		weightUpdate += posPhaseHidden[hiddenId + i] * posPhaseVisible[visibleId + i] - negPhaseHidden[hiddenId + i] * negPhaseVisible[visibleId + i];
	    }

	    weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * weightUpdates[id];
	    weights[id] += weightUpdate;
	    weightUpdates[id] = weightUpdate;
	}
    }

    private static class BiasUpdatesKernel extends Kernel {

	private float[] hiddenBiasWeights;
	private float[] hiddenBiasUpdates;
	private float[] posPhase;
	private float[] negPhase;
	private float learningRate;
	private float momentum;
	private int miniBatchSize;

	public BiasUpdatesKernel(float[] hiddenBiasWeights, float[] posPhase, float[] negPhase, float learningRate, float momentum, int miniBatchSize) {
	    super();
	    this.hiddenBiasWeights = hiddenBiasWeights;
	    this.hiddenBiasUpdates = new float[hiddenBiasWeights.length];
	    this.posPhase = posPhase;
	    this.negPhase = negPhase;
	    this.learningRate = learningRate;
	    this.momentum = momentum;
	    this.miniBatchSize = miniBatchSize;
	}

	@Override
	public void run() {
	    int id = getGlobalId();
	    float weightUpdate = 0;
	    for (int i = 0; i < miniBatchSize; i++) {
		weightUpdate += posPhase[id * miniBatchSize + i] - negPhase[id * miniBatchSize + i];
	    }

	    weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * hiddenBiasUpdates[id];
	    hiddenBiasWeights[id] += weightUpdate;
	    hiddenBiasUpdates[id] = weightUpdate;
	}
    }
}
