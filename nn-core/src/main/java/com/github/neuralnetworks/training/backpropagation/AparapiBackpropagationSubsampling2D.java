package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSubsampling2D;

/**
 * BackPropagation base function for subsampling layers
 */
public class AparapiBackpropagationSubsampling2D extends AparapiSubsampling2D implements BackpropagationConnectionCalculator {

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
    protected Map<Layer, Matrix> activations;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	super.calculate(connections, output, targetLayer);
    }

    @Override
    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	super.init(c, input, output);

	ffActivation = activations.get(c.getInputLayer()).getElements();
    }

    @Override
    public Map<Layer, Matrix> getActivations() {
        return activations;
    }

    @Override
    public void setActivations(Map<Layer, Matrix> activations) {
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
    public float getWeightDecay() {
	// UNUSED
	return 0;
    }

    @Override
    public void setWeightDecay(float weightDecay) {
	// UNUSED
    }
}
