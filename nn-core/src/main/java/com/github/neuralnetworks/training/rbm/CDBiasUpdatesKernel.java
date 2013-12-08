package com.github.neuralnetworks.training.rbm;

import com.amd.aparapi.Kernel;

/**
 * Aparapi kernerl for update of bias updates
 */
public class CDBiasUpdatesKernel extends Kernel {

    /**
     * bias weights
     */
    private float[] biasWeights;

    /**
     * weight updates
     */
    private float[] biasUpdates;

    /**
     * positive phase
     */
    private float[] posPhase;

    /**
     * negative phase
     */
    private float[] negPhase;
    private float learningRate;
    private float momentum;
    private int miniBatchSize;

    public CDBiasUpdatesKernel(float[] hiddenBiasWeights) {
	super();
	this.biasWeights = hiddenBiasWeights;
	this.biasUpdates = new float[hiddenBiasWeights.length];
    }

    public CDBiasUpdatesKernel(float[] hiddenBiasWeights, float[] posPhase, float[] negPhase, float learningRate, float momentum, int miniBatchSize) {
	super();
	this.biasWeights = hiddenBiasWeights;
	this.biasUpdates = new float[hiddenBiasWeights.length];
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

	weightUpdate = learningRate * (weightUpdate / miniBatchSize) + momentum * biasUpdates[id];
	biasWeights[id] += weightUpdate;
	biasUpdates[id] = weightUpdate;
    }

    public float[] getBiasWeights() {
        return biasWeights;
    }

    public void setHiddenBiasWeights(float[] hiddenBiasWeights) {
        this.biasWeights = hiddenBiasWeights;
        if (this.biasUpdates == null || this.biasUpdates.length != hiddenBiasWeights.length) {
            this.biasUpdates = new float[biasUpdates.length];
        }
    }

    public float[] getBiasUpdates() {
        return biasUpdates;
    }

    public void setBiasUpdates(float[] biasUpdates) {
        this.biasUpdates = biasUpdates;
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
