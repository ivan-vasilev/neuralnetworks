package com.github.neuralnetworks.training.rbm;

import com.amd.aparapi.Kernel;

public class CDBiasUpdatesKernel extends Kernel {

    private float[] hiddenBiasWeights;
    private float[] hiddenBiasUpdates;
    private float[] posPhase;
    private float[] negPhase;
    private float learningRate;
    private float momentum;
    private int miniBatchSize;

    public CDBiasUpdatesKernel(float[] hiddenBiasWeights) {
	super();
	this.hiddenBiasWeights = hiddenBiasWeights;
	this.hiddenBiasUpdates = new float[hiddenBiasWeights.length];
    }

    public CDBiasUpdatesKernel(float[] hiddenBiasWeights, float[] posPhase, float[] negPhase, float learningRate, float momentum, int miniBatchSize) {
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

    public float[] getHiddenBiasWeights() {
        return hiddenBiasWeights;
    }

    public void setHiddenBiasWeights(float[] hiddenBiasWeights) {
        this.hiddenBiasWeights = hiddenBiasWeights;
        if (this.hiddenBiasUpdates == null || this.hiddenBiasUpdates.length != hiddenBiasWeights.length) {
            this.hiddenBiasUpdates = new float[hiddenBiasUpdates.length];
        }
    }

    public float[] getHiddenBiasUpdates() {
        return hiddenBiasUpdates;
    }

    public void setHiddenBiasUpdates(float[] hiddenBiasUpdates) {
        this.hiddenBiasUpdates = hiddenBiasUpdates;
    }

    public float[] getPosPhase() {
        return posPhase;
    }

    public void setPosPhase(float[] posPhase) {
        this.posPhase = posPhase;
    }

    public float[] getNegPhase() {
        return negPhase;
    }

    public void setNegPhase(float[] negPhase) {
        this.negPhase = negPhase;
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

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
}
