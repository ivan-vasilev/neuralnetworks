package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * base class for Contrastive Divergence
 */
public abstract class ContrastiveDivergenceTrainer extends OneStepTrainer<RBM> {

    protected Matrix posPhaseVisible;
    protected Matrix negPhaseVisible;
    protected Matrix posPhaseHidden;
    protected Matrix negPhaseHidden;
    protected int miniBatchSize;
    protected ConnectionCalculator hiddenConnectionCalculator;
    protected ConnectionCalculator visibleConnectionCalculator;

    public ContrastiveDivergenceTrainer(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	posPhaseVisible = getPositivePhaseVisible(data);

	if (miniBatchSize != data.getInput().getColumns()) {
	    miniBatchSize = data.getInput().getColumns();
	    init();
	}

	RBMLayerCalculator calculator = (RBMLayerCalculator) getLayerCalculator();

	calculator.calculateHiddenLayer(posPhaseVisible, posPhaseHidden, hiddenConnectionCalculator);

	// Gibbs sampling
	System.arraycopy(posPhaseHidden.getElements(), 0, negPhaseHidden.getElements(), 0, negPhaseHidden.getElements().length);
	int gibbsSamplingCount = properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
	for (int i = 0; i < gibbsSamplingCount; i++) {
	    calculator.calculateVisibleLayer(negPhaseVisible, negPhaseHidden, visibleConnectionCalculator);
	    calculator.calculateHiddenLayer(negPhaseVisible, negPhaseHidden, hiddenConnectionCalculator);
	}

	// update weights
	updateWeights();
    }

    protected Matrix getPositivePhaseVisible(TrainingInputData data) {
	return data.getInput();
    }

    protected void init() {
	hiddenConnectionCalculator = properties.getParameter(Constants.HIDDEN_CONNECTION_CALCULATOR);
	visibleConnectionCalculator = properties.getParameter(Constants.VISIBLE_CONNECTION_CALCULATOR);

	RBM nn = getNeuralNetwork();
	this.negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	this.posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	this.negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
    }

    protected abstract void updateWeights();
}
