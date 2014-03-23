package com.github.neuralnetworks.training.backpropagation;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.util.Util;

/**
 * BackPropagation base function for convolutional layers
 */
public class AparapiBackpropagationConv2D extends AparapiConv2D implements BackPropagationConnectionCalculator {

    private static final long serialVersionUID = -345286029645674230L;

    /**
     * Activation of the output layer from the feedforward phase
     */
    protected float[] ffActivation;

    /**
     * weight updates and momentum
     */
    protected final float[] weightUpdates;
    protected final float[] weightUpdatesMomentum;

    /**
     * BP parameters
     */
    protected float learningRate;
    protected float momentum;
    protected float l1weightDecay;
    protected float l2weightDecay;

    /**
     * activations from the feedforward phase
     */
    protected ValuesProvider activations;

    public AparapiBackpropagationConv2D(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	super(c, valuesProvider, targetLayer);
	this.weightUpdates = new float[c.getWeights().length];
	this.weightUpdatesMomentum = new float[c.getWeights().length];
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	Conv2DConnection c = null;

	for (Connections con : connections) {
	    if (con instanceof Conv2DConnection) {
		c = (Conv2DConnection) con;
	    }
	}

	if (c != null) {
	    Util.fillArray(weightUpdates, 0);

	    if (ffActivation != activations.getValues(c.getInputLayer(), c).getElements()) {
		ffActivation = activations.getValues(c.getInputLayer(), c).getElements();
	    }

	    // currently works only as a feedforward (including bp)
	    if (targetLayer == c.getOutputLayer()) {
		super.calculate(c, valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c), valuesProvider.getValues(targetLayer, c));
	    } else {
		super.calculate(c, valuesProvider.getValues(targetLayer, c), valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c));
	    }

	    updateWeights();
	}
    }

    @Override
    protected void conv(int weightsStartId, int inputStartId, int outputStartId) {
	int id = getGlobalId();

	float activationDerivative = 0;
	int inputId = 0;

	for (int p = 0; p < miniBatchSize; p++) {
	    activationDerivative = activationFunctionDerivative(output[id * miniBatchSize + p]);
	    output[id * miniBatchSize + p] = activationDerivative;

	    for (int i = 0; i < featureMapWeights; i++) {
		inputId = (inputStartId + featureMapOffsets[i]) * miniBatchSize + p;
		weightUpdates[weightsStartId + i] += activationDerivative * ffActivation[inputId];
		input[inputId] += activationDerivative * weights[weightsStartId + i];
	    }
	}
    }

    /**
     * Weight updates after the backpropagation
     */
    protected void updateWeights() {
	float weightUpdate = 0;
	for (int i = 0; i < weights.length; i++) {
	    weightUpdate = learningRate * weightUpdates[i] + momentum * weightUpdatesMomentum[i] - l1weightDecay * Math.abs(weights[i]) - l2weightDecay * weights[i] * weights[i] / 2;
	    weights[i] += weightUpdate;
	    weightUpdatesMomentum[i] = weightUpdates[i];
	    weightUpdates[i] = weightUpdate;
	}
    }

    /**
     * Derivative of the FF activation function
     * 
     * @param value
     * @return
     */
    protected float activationFunctionDerivative(float value) {
	return value;
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
    public float getL1weightDecay() {
        return l1weightDecay;
    }

    @Override
    public void setL1weightDecay(float weightDecay) {
	this.l1weightDecay = weightDecay;
    }
    
    @Override
    public float getL2weightDecay() {
	return l2weightDecay;
    }
    
    @Override
    public void setL2weightDecay(float weightDecay) {
	this.l2weightDecay = weightDecay;
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
