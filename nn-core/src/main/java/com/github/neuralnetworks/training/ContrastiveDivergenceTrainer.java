package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * 
 * base class for contrastive divergence
 *
 */
public abstract class ContrastiveDivergenceTrainer extends Trainer<RBM> {

    private Matrix posPhaseVisible;
    private Matrix negPhaseVisible;
    private Matrix posPhaseHidden;
    private Matrix negPhaseHidden;
    private int miniBatchSize;
    private RBMLayerCalculator calculator;
    private ConnectionCalculator hiddenConnectionCalculator;
    private ConnectionCalculator visibleConnectionCalculator;

    public ContrastiveDivergenceTrainer() {
	super();
    }

    public ContrastiveDivergenceTrainer(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	posPhaseVisible = data.getInput();

	if (miniBatchSize != data.getInput().getColumns()) {
	    miniBatchSize = data.getInput().getColumns();
	    init();
	}

	calculator.calculateHiddenLayer(posPhaseVisible, posPhaseHidden, hiddenConnectionCalculator);

	// Gibbs sampling
	int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
	for (int i = 0; i < gibbsSamplingCount; i++) {
	    calculator.calculateVisibleLayer(negPhaseVisible, posPhaseHidden, visibleConnectionCalculator);
	    calculator.calculateHiddenLayer(negPhaseVisible, negPhaseHidden, hiddenConnectionCalculator);
	}

	// update weights
	updateWeights();
    }

    protected void init() {
	hiddenConnectionCalculator = properties.getParameter(Constants.HIDDEN_CONNECTION_CALCULATOR);
	visibleConnectionCalculator = properties.getParameter(Constants.VISIBLE_CONNECTION_CALCULATOR);

	RBM nn = getNeuralNetwork();
	calculator = new RBMLayerCalculator(nn);

	//this.posPhaseVisible = new Matrix(new float[nn.getVisibleLayer().getNeuronCount()], miniBatchSize);
	this.negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	this.posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	this.negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
    }

    protected abstract void updateWeights();
}
