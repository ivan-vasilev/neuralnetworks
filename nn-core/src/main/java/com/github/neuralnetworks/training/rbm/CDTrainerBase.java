package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * base class for Contrastive Divergence
 */
public abstract class CDTrainerBase extends OneStepTrainer<RBM> {

    private Matrix posPhaseVisible;
    private Matrix negPhaseVisible;
    private Matrix posPhaseHidden;
    private Matrix negPhaseHidden;
    private int miniBatchSize;

    public CDTrainerBase(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	posPhaseVisible = data.getInput();

	if (miniBatchSize != data.getInput().getColumns()) {
	    miniBatchSize = data.getInput().getColumns();

	    RBM nn = getNeuralNetwork();
	    this.negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	    this.posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	    this.negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	}

	triggerEvent(new SamplingStepEvent(this, -1));

	RBMLayerCalculator calculator = (RBMLayerCalculator) getLayerCalculator();

	calculator.calculateHiddenLayer(posPhaseVisible, posPhaseHidden, getHiddenConnectionCalculator(0));

	triggerEvent(new SamplingStepEvent(this, 0));

	// Gibbs sampling
	for (int i = 1; i <= getGibbsSamplingCount(); i++) {
	    calculator.calculateVisibleLayer(negPhaseVisible, negPhaseHidden, getVisibleConnectionCalculator(i));
	    calculator.calculateHiddenLayer(negPhaseVisible, negPhaseHidden, getHiddenConnectionCalculator(i));
	    triggerEvent(new SamplingStepEvent(this, i));
	}

	// update weights
	updateWeights(posPhaseVisible, posPhaseHidden, negPhaseVisible, negPhaseHidden);
    }

    public Matrix getPosPhaseVisible() {
        return posPhaseVisible;
    }

    public Matrix getNegPhaseVisible() {
        return negPhaseVisible;
    }

    public Matrix getPosPhaseHidden() {
        return posPhaseHidden;
    }

    public Matrix getNegPhaseHidden() {
        return negPhaseHidden;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    protected ConnectionCalculator getVisibleConnectionCalculator(int samplingStep) {
	return properties.getParameter(Constants.VISIBLE_CONNECTION_CALCULATOR);
    }

    protected ConnectionCalculator getHiddenConnectionCalculator(int samplingStep) {
	return properties.getParameter(Constants.HIDDEN_CONNECTION_CALCULATOR);
    }

    public int getGibbsSamplingCount() {
	return properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
    }

    protected abstract void updateWeights(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden);

    public static class SamplingStepEvent extends TrainingEvent {

	private static final long serialVersionUID = 1772155171480490374L;

	private int samplingCount;

	public SamplingStepEvent(CDTrainerBase source, int samplingCount) {
	    super(source);
	    this.samplingCount = samplingCount;
	}

	public int getSamplingCount() {
	    return samplingCount;
	}
    }
}
