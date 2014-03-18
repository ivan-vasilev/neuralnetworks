package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSubsampling2D;

/**
 * BackPropagation base function for subsampling layers
 */
public class AparapiBackpropagationSubsampling2D extends AparapiSubsampling2D implements BackPropagationConnectionCalculator {

    private static final long serialVersionUID = -345286029645674230L;

    public AparapiBackpropagationSubsampling2D(Subsampling2DConnection c, int miniBatchSize) {
	super(c, miniBatchSize);
    }

    /**
     * Activation of the output layer from the feedforward phase
     */
    protected float[] ffActivation;

    /**
     * activations from the feedforward phase
     */
    protected ValuesProvider activations;

    @Override
    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	super.init(c, input, output);

	ffActivation = activations.getValues(c.getInputLayer(), c).getElements();
    }

    @Override
    public ValuesProvider getActivations() {
        return activations;
    }

    @Override
    public void setActivations(ValuesProvider activations) {
        this.activations = activations;
    }

    @Override
    public float getLearningRate() {
	// UNUSED
	return 0;
    }

    @Override
    public void setLearningRate(float learningRate) {
	// UNUSED
    }

    @Override
    public float getMomentum() {
	// UNUSED
	return 0;
    }

    @Override
    public void setMomentum(float momentum) {
	// UNUSED
    }

    @Override
    public float getL1weightDecay() {
	// UNUSED
	return 0;
    }

    @Override
    public void setL1weightDecay(float weightDecay) {
	// UNUSED
    }

    @Override
    public float getL2weightDecay() {
	// UNUSED
	return 0;
    }

    @Override
    public void setL2weightDecay(float l2weightDecay) {
	// UNUSED
    }
}
