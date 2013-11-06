package com.github.neuralnetworks.training.rbm;

import com.amd.aparapi.Kernel;

public class CDWeightUpdatesKernel extends Kernel {

    private float[] posPhaseHidden;
    private float[] posPhaseVisible;
    private float[] negPhaseHidden;
    private float[] negPhaseVisible;
    private float[] weights;
    private float[] weightUpdates;
    private int weightColumns;
    private float learningRate;
    private float momentum;
    private float weightDecay;
    private int miniBatchSize;

    public CDWeightUpdatesKernel(float[] weights, int weightColumns) {
	super();
	this.weights = weights;
	this.weightUpdates = new float[weights.length];
	this.weightColumns = weightColumns;
    }

    public CDWeightUpdatesKernel(float[] posPhaseVisible, float[] posPhaseHidden, float[] negPhaseVisible, float[] negPhaseHidden, float[] weights, int weightColumns, float learningRate, float momentum, float weightDecay, int miniBatchSize) {
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
	this.weightDecay = weightDecay;
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

	weightUpdate = learningRate * (weightUpdate / miniBatchSize - weightDecay * weights[id]) + momentum * weightUpdates[id];
	weights[id] += weightUpdate;
	weightUpdates[id] = weightUpdate;
    }

    public float[] getPosPhaseHidden() {
        return posPhaseHidden;
    }

    public void setPosPhaseHidden(float[] posPhaseHidden) {
        this.posPhaseHidden = posPhaseHidden;
    }

    public float[] getPosPhaseVisible() {
        return posPhaseVisible;
    }

    public void setPosPhaseVisible(float[] posPhaseVisible) {
        this.posPhaseVisible = posPhaseVisible;
    }

    public float[] getNegPhaseHidden() {
        return negPhaseHidden;
    }

    public void setNegPhaseHidden(float[] negPhaseHidden) {
        this.negPhaseHidden = negPhaseHidden;
    }

    public float[] getNegPhaseVisible() {
        return negPhaseVisible;
    }

    public void setNegPhaseVisible(float[] negPhaseVisible) {
        this.negPhaseVisible = negPhaseVisible;
    }

    public float[] getWeights() {
        return weights;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
        if (this.weightUpdates == null || this.weightUpdates.length != weights.length) {
            this.weightUpdates = new float[weights.length];
        }
    }

    public int getWeightColumns() {
        return weightColumns;
    }

    public void setWeightColumns(int weightColumns) {
        this.weightColumns = weightColumns;
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

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public float getWeightDecay() {
        return weightDecay;
    }

    public void setWeightDecay(float weightDecay) {
        this.weightDecay = weightDecay;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
}
