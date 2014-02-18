package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    protected ValuesProvider activations;

    public AparapiBackpropagationFullyConnected(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	super(inputConnections, miniBatchSize, targetLayer);
    }

    @Override
    public void calculate(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super.calculate(inputConnections, valuesProvider, targetLayer);

	if (inputConnections.size() > 2 || (inputConnections.size() > 1 && !Util.hasBias(inputConnections))) {
	    int i = 0;
	    for (Connections c : inputConnections) {
		float[] a = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c).getElements();
		System.arraycopy(input, inputStartPositions[i], a, 0, a.length);
		float[] cg = ((GraphConnections) c).getConnectionGraph().getElements();
		System.arraycopy(weights, weightStartPositions[i], cg, 0, cg.length);
		i++;
	    }
	}
    }

    @Override
    protected void init(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super.init(inputConnections, valuesProvider, targetLayer);
	weightUpdates = storedWeightUpdates.get(targetLayer);
	if (weightUpdates == null) {
	    weightUpdates = new float[weights.length];
	    storedWeightUpdates.put(targetLayer, weightUpdates);
	}

	ffActivation = activations.getValues(targetLayer, inputConnections).getElements();
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
		weightUpdate = lr * weightUpdate + mm * weightUpdates[weightIndex] - wd * abs(weights[weightIndex]);
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
    public ValuesProvider getActivations() {
	return activations;
    }

    @Override
    public void setActivations(ValuesProvider activations) {
	this.activations = activations;
    }
}
