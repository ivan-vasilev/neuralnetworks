package com.github.neuralnetworks.training.rbm;

import com.amd.aparapi.Kernel;

/**
 * Aparapi kernerl for update of bias updates
 */
public class CDBiasUpdatesKernel extends Kernel {

    /**
     * bias weights
     */
    private final float[] biasWeights;

    /**
     * weight updates
     */
    private final float[] biasUpdates;

    /**
     * positive phase
     */
    private final float[] posPhase;

    /**
     * negative phase
     */
    private final float[] negPhase;
    private float learningRate;
    private final float momentum;
    private final int miniBatchSize;

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

	weightUpdate = learningRate * weightUpdate /*/ mbs */ + momentum * biasUpdates[id];
	biasWeights[id] += weightUpdate;
	biasUpdates[id] = weightUpdate;
    }

    public float[] getBiasWeights() {
        return biasWeights;
    }

    public float[] getBiasUpdates() {
        return biasUpdates;
    }

    public float[] getPosPhase() {
        return posPhase;
    }

    public float[] getNegPhase() {
        return negPhase;
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

    public int getMiniBatchSize() {
        return miniBatchSize;
    }
}
