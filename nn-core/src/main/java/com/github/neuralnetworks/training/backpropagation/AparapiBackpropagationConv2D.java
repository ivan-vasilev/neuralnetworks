package com.github.neuralnetworks.training.backpropagation;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Tensor.TensorIterator;
import com.github.neuralnetworks.util.TensorFactory;
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
    protected final int activationStartIndex;
    protected final int activationFeatureMapRowsDistance;
    protected final int activationFeatureMapColumnsDistance;

    /**
     * weight updates and momentum
     */
    protected final Tensor weightUpdatesTensor;
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

    public AparapiBackpropagationConv2D(Conv2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates, Layer targetLayer) {
	super(c, valuesProvider, targetLayer);

	Tensor t = TensorFactory.tensor(targetLayer, c, activations);
	this.ffActivation = t.getElements();
	this.activationStartIndex = t.getStartIndex();
	this.activationFeatureMapRowsDistance = t.getDimensionElementsDistance(1);
	this.activationFeatureMapColumnsDistance = t.getDimensionElementsDistance(2);

	this.weightUpdatesTensor = weightUpdates;
	this.weightUpdates = weightUpdates.getElements();
	this.weightUpdatesMomentum = new float[c.getWeights().getSize()];
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
	    weightUpdatesTensor.forEach(i -> weightUpdates[i] = 0);

	    // currently works only as a feedforward (including bp)
	    if (targetLayer == c.getOutputLayer()) {
		super.calculate(c, valuesProvider, targetLayer);
	    } else {
		super.calculate(c, valuesProvider, Util.getOppositeLayer(c, targetLayer));
	    }

	    updateWeights();
	}
    }

    @Override
    protected void conv(int weightsStartId, int inputStartId, int outputStartId) {
	float activationDerivative = 0;
	int activationStartId = activationStartIndex + ((getGlobalId() % outputFeatureMapLength) / outputColumns) * activationFeatureMapRowsDistance * stride + (getGlobalId() % outputColumns) * activationFeatureMapColumnsDistance * stride;

	for (int i = 0; i < miniBatchSize; i++) {
	    activationDerivative = activationFunctionDerivative(output[outputStartId + i * outputMiniBatchDistance]);
	    output[outputStartId + i * outputMiniBatchDistance] = activationDerivative;

	    for (int j = 0; j < featureMapWeights; j++) {
		weightUpdates[weightsStartId + j] += activationDerivative * ffActivation[activationStartId + featureMapOffsets[i * featureMapWeights + j]];
		input[inputStartId + featureMapOffsets[i * featureMapWeights + j]] += activationDerivative * weights[weightsStartId + j];
	    }
	}
    }

    /**
     * Weight updates after the backpropagation
     */
    protected void updateWeights() {
	float weightUpdate = 0;
	TensorIterator it = weightUpdatesTensor.iterator();
	for (int i = 0, j = 0; it.hasNext(); j++) {
	    i = it.next();
	    weightUpdate = learningRate * weightUpdates[i] + momentum * weightUpdatesMomentum[j] - l1weightDecay * Math.abs(weights[i]) - l2weightDecay * weights[i] * weights[i] / 2;
	    weights[i] += weightUpdate;
	    weightUpdatesMomentum[j] = weightUpdates[i];
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
