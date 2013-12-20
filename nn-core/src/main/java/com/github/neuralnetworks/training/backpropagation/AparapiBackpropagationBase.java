package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;

/**
 * Aparapi Backpropagation base weighted sum
 * Supports learning rate, momentum and weight decay
 */
public class AparapiBackpropagationBase extends AparapiWeightedSum {

    private static final long serialVersionUID = -5101971690861270462L;

    /**
     * Activation of the output layer from the feedforward phase
     */
    protected float[] outputActivation;

    /**
     * Weight updates array
     */
    protected float[] weightUpdates;

    /**
     * Stored weight updates for reuse
     */
    protected Map<Layer, float[]> storedWeightUpdates = new HashMap<>();
    protected float learningRate;
    protected float momentum;
    protected float weightDecay;

    /**
     * activations from the feedforward phase
     */
    protected Map<Layer, Matrix> activations;

    @Override
    public void calculate(SortedMap<Connections, Matrix> inputConnections, Matrix outputMatrix, Layer targetLayer) {
	super.calculate(inputConnections, outputMatrix, targetLayer);
	if (inputConnections.size() > 1) {
	    int i = 0;
	    for (java.util.Map.Entry<Connections, Matrix> e : inputConnections.entrySet()) {
		System.arraycopy(input, inputStartPositions[i], e.getValue().getElements(), 0, e.getValue().getElements().length);
		float[] cg = ((GraphConnections) e.getKey()).getConnectionGraph().getElements();
		System.arraycopy(weights, weightStartPositions[i], cg, 0, cg.length);
		i++;
	    }
	}
    }

    @Override
    protected void init(SortedMap<GraphConnections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	super.init(input, outputMatrix, targetLayer);

	weightUpdates = storedWeightUpdates.get(targetLayer);
	if (weightUpdates == null) {
	    weightUpdates = new float[weights.length];
	    storedWeightUpdates.put(targetLayer, weightUpdates);
	}

	outputActivation = activations.get(targetLayer).getElements();
    }

    @Override
    protected void after(int row, int column) {
	calcDerivativeBefore(row, column);

	int id = getGlobalId();

	for (int i = 0; i < series; i++) {
	    int inputStartPosition = inputStartPositions[i];
	    int inputStartIndex = inputStartIndexes[i];
	    int initialWeightIndex = weightStartPositions[i] + weightsInitialStep[i] * id;
	    int weightStep = weightsStep[i];
	    int dim = weightsDimension[i];
	    int outputIndex = outputIndex(row, column, i);

	    for (int j = 0; j < dim; j++) {
		int weightIndex = initialWeightIndex + j * weightStep;
		float weightUpdate = learningRate * (input[inputStartPosition + (inputStartIndex + j) * inputOutputColumns + column] * outputActivation[outputIndex] - weightDecay * weights[weightIndex]) + momentum * weightUpdates[weightIndex];
		weights[weightIndex] += weightUpdate;
		weightUpdates[weightIndex] = weightUpdate;
	    }
	}

	calcDerivativeAfter(row, column);
    }

    /**
     * calculate derivative before weights update
     */
    protected void calcDerivativeBefore(int row, int column) {
    }

    /**
     * calculate derivative after weights update
     */
    protected void calcDerivativeAfter(int row, int column) {
    }

    public float[] getOutputActivation() {
        return outputActivation;
    }

    public void setOutputActivation(float[] outputActivation) {
        this.outputActivation = outputActivation;
    }

    public float[] getWeightUpdates() {
        return weightUpdates;
    }

    public void setWeightUpdates(float[] weightUpdates) {
        this.weightUpdates = weightUpdates;
    }

    public Map<Layer, float[]> getStoredWeightUpdates() {
        return storedWeightUpdates;
    }

    public void setStoredWeightUpdates(Map<Layer, float[]> storedWeightUpdates) {
        this.storedWeightUpdates = storedWeightUpdates;
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

    public Map<Layer, Matrix> getActivations() {
        return activations;
    }

    public void setActivations(Map<Layer, Matrix> activations) {
        this.activations = activations;
    }
}
