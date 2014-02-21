package com.github.neuralnetworks.training.backpropagation;

import java.util.List;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
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
    protected final float[] weightUpdates;

    protected float learningRate;
    protected final float momentum;
    protected final float l1weightDecay;
    protected final float l2weightDecay;

    /**
     * activations from the feedforward phase
     */
    protected ValuesProvider activations;

    public AparapiBackpropagationFullyConnected(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, Layer targetLayer) {
	super(inputConnections, miniBatchSize, targetLayer);
	this.learningRate = momentum;
	this.momentum = momentum;
	this.l1weightDecay = l1weightDecay;
	this.l2weightDecay = l2weightDecay;
	this.weightUpdates = new float[weights.length];
    }

    @Override
    public void calculate(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super.calculate(inputConnections, valuesProvider, targetLayer);

	if (inputConnections.size() > 2 || (inputConnections.size() > 1 && !Util.hasBias(inputConnections))) {
	    int i = 0;
	    for (Connections c : inputConnections) {
		float[] a = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c).getElements();
		System.arraycopy(a, inputStartPositions[i], input, 0, a.length);
		float[] cg = ((GraphConnections) c).getConnectionGraph().getElements();
		System.arraycopy(cg, weightStartPositions[i], weights, 0, cg.length);
		i++;
	    }
	}
    }

    @Override
    protected void init(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super.init(inputConnections, valuesProvider, targetLayer);
	if (ffActivation != activations.getValues(targetLayer, inputConnections).getElements()) {
	    ffActivation = activations.getValues(targetLayer, inputConnections).getElements();
	}
    }

    @Override
    protected void after() {
	final int row = getGlobalId() * miniBatchSize;
	float lr = learningRate;
	float weight = 0, weightUpdate = 0;
	int inputStartPosition = 0, initialWeightIndex = 0, weightStep = 0, dim = 0, weightIndex = 0;

	for (int i = 0; i < series; i++) {
	    inputStartPosition = inputStartPositions[i];
	    initialWeightIndex = weightStartPositions[i] + weightsInitialStep[i] * getGlobalId();
	    weightStep = weightsStep[i];
	    dim = weightsDimension[i];

	    for (int j = 0; j < dim; j++) {
		weightUpdate = 0;
		for (int column = 0; column < miniBatchSize; column++) {
		    weightUpdate += input[inputStartPosition + j * miniBatchSize + column] * ffActivation[row + column];
		}

		weightIndex = initialWeightIndex + j * weightStep;
		weight = weights[weightIndex];
		weightUpdate = lr * weightUpdate + momentum * weightUpdates[weightIndex] - l1weightDecay * abs(weight) - l2weightDecay * weight * weight / 2;
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
    }

    @Override
    public float getL1weightDecay() {
	return l1weightDecay;
    }

    @Override
    public void setL1weightDecay(float weightDecay) {
    }

    @Override
    public float getL2weightDecay() {
	return l2weightDecay;
    }

    @Override
    public void setL2weightDecay(float l2weightDecay) {
    }

    @Override
    public ValuesProvider getActivations() {
	return activations;
    }

    @Override
    public void setActivations(ValuesProvider activations) {
	this.activations = activations;
    }
}
