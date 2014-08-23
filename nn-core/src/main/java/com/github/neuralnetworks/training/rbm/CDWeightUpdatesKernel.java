package com.github.neuralnetworks.training.rbm;

import java.io.Serializable;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.tensor.Matrix;

/**
 * Aparapi weight udpates for the connections between the hidden and the visible layers
 */
public class CDWeightUpdatesKernel extends Kernel implements Serializable {

    private static final long serialVersionUID = 1L;

    // data parameters
    private final float[] posPhaseVisible;
    private final int posPhaseVisibleStartIndex;
    private final int posPhaseVisibleRowStep;
    private final int posPhaseVisibleColumnStep;

    private final float[] posPhaseHidden;
    private final int posPhaseHiddenStartIndex;
    private final int posPhaseHiddenRowStep;
    private final int posPhaseHiddenColumnStep;

    private final float[] negPhaseVisible;
    private final int negPhaseVisibleStartIndex;
    private final int negPhaseVisibleRowStep;
    private final int negPhaseVisibleColumnStep;

    private final float[] negPhaseHidden;
    private final int negPhaseHiddenStartIndex;
    private final int negPhaseHiddenRowStep;
    private final int negPhaseHiddenColumnStep;

    private final int miniBatchSize;

    // weights parameters
    private final float[] weights;
    private final int weightsStartIndex;
    private final int weightsRowStep;
    private final int weightsColumnStep;
    private final int weightsColumns;
    private final float[] weightUpdates;

    // learning parameters
    private float learningRate;
    private final float momentum;
    private final float l1weightDecay;
    private final float l2weightDecay;

    public CDWeightUpdatesKernel(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden, Matrix weights, float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
	super();
	this.posPhaseVisible = posPhaseVisible.getElements();
	this.posPhaseVisibleStartIndex = posPhaseVisible.getStartIndex();
	this.posPhaseVisibleRowStep = posPhaseVisible.getRowElementsDistance();
	this.posPhaseVisibleColumnStep = posPhaseVisible.getColumnElementsDistance();

	this.posPhaseHidden = posPhaseHidden.getElements();
	this.posPhaseHiddenStartIndex = posPhaseHidden.getStartIndex();
	this.posPhaseHiddenRowStep = posPhaseHidden.getRowElementsDistance();
	this.posPhaseHiddenColumnStep = posPhaseHidden.getColumnElementsDistance();

	this.negPhaseVisible = negPhaseVisible.getElements();
	this.negPhaseVisibleStartIndex = negPhaseVisible.getStartIndex();
	this.negPhaseVisibleRowStep = negPhaseVisible.getRowElementsDistance();
	this.negPhaseVisibleColumnStep = negPhaseVisible.getColumnElementsDistance();

	this.negPhaseHidden = negPhaseHidden.getElements();
	this.negPhaseHiddenStartIndex = negPhaseHidden.getStartIndex();
	this.negPhaseHiddenRowStep = negPhaseHidden.getRowElementsDistance();
	this.negPhaseHiddenColumnStep = negPhaseHidden.getColumnElementsDistance();

	this.weights = weights.getElements();
	this.weightsStartIndex = weights.getStartIndex();
	this.weightsRowStep = weights.getRowElementsDistance();
	this.weightsColumnStep = weights.getColumnElementsDistance();
	this.weightsColumns = weights.getColumns();

	this.weightUpdates = new float[weights.getSize()];
	this.learningRate = learningRate;
	this.momentum = momentum;
	this.l1weightDecay = l1weightDecay;
	this.l2weightDecay = l2weightDecay;
	this.miniBatchSize = posPhaseVisible.getColumns();
    }

    @Override
    public void run() {
	int id = getGlobalId();

	int weightId = 0, weightUpdateId = 0;
	float weightUpdate = 0, weight = 0;

	for (int i = 0; i < weightsColumns; i++) {
	    weightUpdate = 0;

	    for (int j = 0; j < miniBatchSize; j++) {
		weightUpdate += posPhaseHidden[posPhaseHiddenStartIndex + id * posPhaseHiddenRowStep + j * posPhaseHiddenColumnStep] * posPhaseVisible[posPhaseVisibleStartIndex + i * posPhaseVisibleRowStep + j * posPhaseVisibleColumnStep] - negPhaseHidden[negPhaseHiddenStartIndex + id * negPhaseHiddenRowStep + j * negPhaseHiddenColumnStep] * negPhaseVisible[negPhaseVisibleStartIndex + i * negPhaseVisibleRowStep + j * negPhaseVisibleColumnStep];
	    }

	    weightId = weightsStartIndex + id * weightsRowStep + i * weightsColumnStep;
	    weightUpdateId = id * weightsColumns + i;
	    weight = weights[weightId];
	    weightUpdate = learningRate * (weightUpdate /*/ mbs*/ - l1weightDecay * abs(weight) - l2weightDecay * weight * weight / 2) + momentum * weightUpdates[weightUpdateId];
	    weights[weightId] += weightUpdate;
	    weightUpdates[weightUpdateId] = weightUpdate;
	}
    }

    public float[] getPosPhaseHidden() {
        return posPhaseHidden;
    }

    public float[] getPosPhaseVisible() {
        return posPhaseVisible;
    }

    public float[] getNegPhaseHidden() {
        return negPhaseHidden;
    }

    public float[] getNegPhaseVisible() {
        return negPhaseVisible;
    }

    public float[] getWeights() {
        return weights;
    }

    public int getWeightColumns() {
        return weightsColumns;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getMomentum() {
        return momentum;
    }

    public float getl1WeightDecay() {
        return l1weightDecay;
    }
    
    public float getl2WeightDecay() {
	return l1weightDecay;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }
}
