package com.github.neuralnetworks.training.rbm;

import com.amd.aparapi.Kernel;

/**
 * Aparapi weight udpates for the connections between the hidden and the visible layers
 */
public class CDWeightUpdatesKernel extends Kernel {

    private final float[] posPhaseHidden;
    private final float[] posPhaseVisible;
    private final float[] negPhaseHidden;
    private final float[] negPhaseVisible;
    private final float[] weights;
    private final float[] weightUpdates;
    private final int weightColumns;
    private final int miniBatchSize;
    private float learningRate;
    private final float momentum;
    private final float l1weightDecay;
    private final float l2weightDecay;

    public CDWeightUpdatesKernel(float[] posPhaseVisible, float[] posPhaseHidden, float[] negPhaseVisible, float[] negPhaseHidden, float[] weights, int weightColumns, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int miniBatchSize) {
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
	this.l1weightDecay = l1weightDecay;
	this.l2weightDecay = l2weightDecay;
	this.miniBatchSize = miniBatchSize;
    }

    @Override
    public void run() {
	int id = getGlobalId();
	int mbs = miniBatchSize;
	int wc = weightColumns;
	int hiddenId = id * mbs;
	float lr = learningRate;
	float mm = momentum;

	int visibleId = 0, weightId = 0;
	float weightUpdate = 0, weight = 0;

	for (int i = 0; i < wc; i++) {
	    visibleId = i * mbs;
	    weightUpdate = 0;

	    for (int j = 0; j < mbs; j++) {
		weightUpdate += posPhaseHidden[hiddenId + j] * posPhaseVisible[visibleId + j] - negPhaseHidden[hiddenId + j] * negPhaseVisible[visibleId + j];
	    }

	    weightId = id * wc + i;
	    weight = weights[weightId];
	    weightUpdate = lr * (weightUpdate /*/ mbs*/ - l1weightDecay * abs(weight) - l2weightDecay * weight * weight / 2) + mm * weightUpdates[weightId];
	    weights[weightId] += weightUpdate;
	    weightUpdates[weightId] = weightUpdate;
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
        return weightColumns;
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
