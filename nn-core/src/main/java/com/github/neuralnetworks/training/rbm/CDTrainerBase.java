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
 * Base class for Contrastive Divergence
 * requires RBMLayerCalculator as the layer calculator. This allows for different implementations of the layer calculator, like GPU/CPU for example
 */
public abstract class CDTrainerBase extends OneStepTrainer<RBM> {

    /**
     * positive phase visible layer results
     */
    private Matrix posPhaseVisible;

    /**
     * negative phase visible layer results
     */
    private Matrix negPhaseVisible;

    /**
     * positive phase hidden layer results
     */
    private Matrix posPhaseHidden;

    /**
     * negative phase hidden layer results
     */
    private Matrix negPhaseHidden;

    /**
     * size of the mini batch
     */
    private int miniBatchSize;

    public CDTrainerBase(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	posPhaseVisible = data.getInput();

	if (miniBatchSize != posPhaseVisible.getColumns()) {
	    miniBatchSize = posPhaseVisible.getColumns();

	    RBM nn = getNeuralNetwork();
	    this.negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	    this.posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	    this.negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	}

	triggerEvent(new SamplingStepEvent(this, -1));

	RBMLayerCalculator calculator = (RBMLayerCalculator) getLayerCalculator();

	// calculate hidden layer positive phase
	calculator.calculateHiddenLayer(getNeuralNetwork(), posPhaseVisible, posPhaseHidden, getHiddenConnectionCalculator(0));

	triggerEvent(new SamplingStepEvent(this, 0));

	// Gibbs sampling
	for (int i = 1; i <= getGibbsSamplingCount(); i++) {
	    calculator.calculateVisibleLayer(getNeuralNetwork(), negPhaseVisible, negPhaseHidden, getVisibleConnectionCalculator(i));
	    calculator.calculateHiddenLayer(getNeuralNetwork(), negPhaseVisible, negPhaseHidden, getHiddenConnectionCalculator(i));
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
