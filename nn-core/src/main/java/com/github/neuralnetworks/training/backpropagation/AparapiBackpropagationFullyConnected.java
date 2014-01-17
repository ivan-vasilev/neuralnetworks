package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;
import com.github.neuralnetworks.util.Util;

/**
 * Aparapi Backpropagation base weighted sum Supports learning rate, momentum
 * and weight decay
 */
public class AparapiBackpropagationFullyConnected extends AparapiWeightedSum implements BackpropagationConnectionCalculator {

    private static final long serialVersionUID = -5101971690861270462L;

    /**
     * Activation of the output layer from the feedforward phase
     */
    protected float[] ffActivation;

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

    public AparapiBackpropagationFullyConnected(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	super(inputConnections, miniBatchSize, targetLayer);
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> inputConnections, Matrix outputMatrix, Layer targetLayer) {
	super.calculate(inputConnections, outputMatrix, targetLayer);

	if (inputConnections.size() > 2 || (inputConnections.size() > 1 && !Util.hasBias(inputConnections.keySet()))) {
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

	ffActivation = activations.get(targetLayer).getElements();
    }

    @Override
    protected void after() {
	int s = series;
	int miniBatch = miniBatchSize;
	int row = getGlobalId() * miniBatch;
	float lr = learningRate;
	float wd = weightDecay;
	float mm = momentum;
	float weightUpdate = 0;
	int inputStartPosition = 0, initialWeightIndex = 0, weightStep = 0, dim = 0, weightIndex = 0;

	for (int i = 0; i < s; i++) {
	    inputStartPosition = inputStartPositions[i];
	    initialWeightIndex = weightStartPositions[i] + weightsInitialStep[i] * getGlobalId();
	    weightStep = weightsStep[i];
	    dim = weightsDimension[i];

	    for (int j = 0; j < dim; j++) {
		weightUpdate = 0;
		for (int column = 0; column < miniBatch; column++) {
		    weightUpdate += input[inputStartPosition + j * miniBatch + column] * ffActivation[row + column];
		}

		weightIndex = initialWeightIndex + j * weightStep;
		weightUpdate /= miniBatch;
		weightUpdate = lr * (weightUpdate / miniBatch) - wd * weights[weightIndex] + mm * weightUpdates[weightIndex];
		weights[weightIndex] += weightUpdate;
		weightUpdates[weightIndex] = weightUpdate;
	    }
	}

	calcDerivative();
    }

    /**
     * calculate derivative after weights update
     */
    protected void calcDerivative() {
    }

    @Override
    public float getLearningRate() {
	return learningRate;
    }

    @Override
    public void setLearningRate(float learningRate) {
	this.learningRate = learningRate;
    }

    @Override
    public float getMomentum() {
	return momentum;
    }

    @Override
    public void setMomentum(float momentum) {
	this.momentum = momentum;
    }

    @Override
    public float getWeightDecay() {
	return weightDecay;
    }

    @Override
    public void setWeightDecay(float weightDecay) {
	this.weightDecay = weightDecay;
    }

    @Override
    public Map<Layer, Matrix> getActivations() {
	return activations;
    }

    @Override
    public void setActivations(Map<Layer, Matrix> activations) {
	this.activations = activations;
    }
}
