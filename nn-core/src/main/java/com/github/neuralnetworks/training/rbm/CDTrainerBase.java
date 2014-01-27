package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
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
    protected void learnInput(TrainingInputData data, int batch) {
	RBM nn = getNeuralNetwork();

	posPhaseVisible = data.getInput();
	if (miniBatchSize != posPhaseVisible.getColumns()) {
	    miniBatchSize = posPhaseVisible.getColumns();
	    negPhaseVisible = new Matrix(nn.getVisibleLayer().getNeuronCount(), miniBatchSize);
	    posPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	    negPhaseHidden = new Matrix(nn.getHiddenLayer().getNeuronCount(), miniBatchSize);
	}

	getLayerCalculator().gibbsSampling(nn, posPhaseVisible, posPhaseHidden, negPhaseVisible, negPhaseHidden, getGibbsSamplingCount(), batch == 0 ? true : getResetRBM());

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

    public RBMLayerCalculator getLayerCalculator() {
	return properties.getParameter(Constants.LAYER_CALCULATOR);
    }

    public void setLayerCalculator(RBMLayerCalculator layerCalculator) {
	properties.setParameter(Constants.LAYER_CALCULATOR, layerCalculator);
    }
    
    public Boolean getResetRBM() {
	return properties.getParameter(Constants.RESET_RBM);
    }
    
    public void setResetRBM(boolean resetRBM) {
	properties.setParameter(Constants.RESET_RBM, resetRBM);
    }

    public int getGibbsSamplingCount() {
	return properties.containsKey(Constants.GIBBS_SAMPLING_COUNT) ? (int) properties.get(Constants.GIBBS_SAMPLING_COUNT) : 1;
    }

    protected abstract void updateWeights(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden);
}
