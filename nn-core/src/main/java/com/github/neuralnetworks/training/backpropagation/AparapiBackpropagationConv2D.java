package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.util.Util;

/**
 * BackPropagation base function for convolutional layers
 */
public class AparapiBackpropagationConv2D extends AparapiConv2D implements BackpropagationConnectionCalculator {

    private static final long serialVersionUID = -345286029645674230L;

    /**
     * Activation of the output layer from the feedforward phase
     */
    protected float[] ffActivation;

    /**
     * weight updates and momentum
     */
    protected float[] weightUpdates;
    protected float[] weightUpdatesMomentum;

    /**
     * BP parameters
     */
    protected float learningRate;
    protected float momentum;
    protected float weightDecay;

    /**
     * activations from the feedforward phase
     */
    protected Map<Layer, Matrix> activations;

    public AparapiBackpropagationConv2D(Conv2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = null;
	Conv2DConnection bias = null;

	for (Connections con : connections.keySet()) {
	    if (con instanceof Conv2DConnection) {
		if (c.getInputLayer() instanceof BiasLayer) {
		    bias = (Conv2DConnection) con;
		} else {
		    c = (Conv2DConnection) con;
		}
	    }
	}

	if (bias != null) {
	    
	}

	if (c != null) {
	    // currently works only as a feedforward (including bp)
	    if (targetLayer == c.getOutputLayer()) {
		super.calculate(c, connections.get(c), output);
	    } else {
		super.calculate(c, output, connections.get(c));
	    }

	    updateWeights();
	}
    }

    @Override
    protected void init(Conv2DConnection c, Matrix input, Matrix output) {
	super.init(c, input, output);

	int weightUpdatesLength = c.getWeights().length * output.getColumns();
	if (weightUpdates == null || weightUpdates.length != weightUpdatesLength) {
	    weightUpdates = new float[weightUpdatesLength];
	    weightUpdatesMomentum = new float[weightUpdatesLength];
	} else {
	    Util.fillArray(weightUpdates, 0);
	}

	ffActivation = activations.get(c.getInputLayer()).getElements();
    }

    @Override
    protected void conv(int weightsStartId, int inputStartId) {
	int id = getGlobalId();

	int miniBatch = miniBatchSize;
	int fmw = featureMapWeights;
	float activation = 0;
	int inputId = 0;

	for (int p = 0; p < miniBatch; p++) {
	    activation = activationFunctionDerivative(output[id * miniBatch + p]);;
	    output[id * miniBatch + p] = activation;

	    for (int i = 0, j = weightsStartId; i < fmw; i++, j++) {
		inputId = (inputStartId + featureMapOffsets[i]) * miniBatch + p;
		input[inputId] += activation * weights[j];
		weightUpdates[j * miniBatch + p] += activation * ffActivation[inputId];
	    }
	}
    }

    /**
     * Weight updates after the backpropagation
     */
    protected void updateWeights() {
	float weightUpdate = 0;
	for (int i = 0; i < weights.length; i++) {
	    weightUpdate = learningRate * weightUpdates[i] + momentum * weightUpdatesMomentum[i] - weightDecay * weights[i];
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
