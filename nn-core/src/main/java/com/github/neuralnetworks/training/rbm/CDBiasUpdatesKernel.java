package com.github.neuralnetworks.training.rbm;

import java.io.Serializable;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.tensor.Matrix;

/**
 * Aparapi kernerl for update of bias updates
 */
public class CDBiasUpdatesKernel extends Kernel implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * bias weights
     */
    private final float[] biasWeights;
    private final int weightsStartIndex;
    private final int weightsRowStep;

    /**
     * weight updates
     */
    private final float[] biasUpdates;

    /**
     * positive phase
     */
    private final float[] posPhase;
    private final int posPhaseStartIndex;
    private final int posPhaseRowStep;
    private final int posPhaseColumnStep;

    /**
     * negative phase
     */
    private final float[] negPhase;
    private final int negPhaseStartIndex;
    private final int negPhaseRowStep;
    private final int negPhaseColumnStep;

    private float learningRate;
    private final float momentum;
    private final int miniBatchSize;

    public CDBiasUpdatesKernel(Matrix hiddenBiasWeights, Matrix posPhase, Matrix negPhase, float learningRate, float momentum) {
	super();
	this.posPhase = posPhase.getElements();
	this.posPhaseStartIndex = posPhase.getStartIndex();
	this.posPhaseRowStep = posPhase.getRowElementsDistance();
	this.posPhaseColumnStep = posPhase.getColumnElementsDistance();

	this.negPhase = negPhase.getElements();
	this.negPhaseStartIndex = negPhase.getStartIndex();
	this.negPhaseRowStep = negPhase.getRowElementsDistance();
	this.negPhaseColumnStep = negPhase.getColumnElementsDistance();

	this.biasWeights = hiddenBiasWeights.getElements();
	this.weightsStartIndex = hiddenBiasWeights.getStartIndex();
	this.weightsRowStep = hiddenBiasWeights.getRowElementsDistance();
	this.biasUpdates = new float[hiddenBiasWeights.getSize()];

	this.learningRate = learningRate;
	this.momentum = momentum;
	this.miniBatchSize = posPhase.getColumns();
    }

    @Override
    public void run() {
	int id = getGlobalId();
	float weightUpdate = 0;

	for (int i = 0; i < miniBatchSize; i++) {
	    weightUpdate += posPhase[posPhaseStartIndex + id * posPhaseRowStep + i * posPhaseColumnStep] - negPhase[negPhaseStartIndex + id * negPhaseRowStep + i * negPhaseColumnStep];
	}

	weightUpdate = learningRate * weightUpdate /*/ mbs */ + momentum * biasUpdates[id];
	biasWeights[weightsStartIndex + id * weightsRowStep] += weightUpdate;
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
