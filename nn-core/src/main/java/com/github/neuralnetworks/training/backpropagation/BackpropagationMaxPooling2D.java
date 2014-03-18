package com.github.neuralnetworks.training.backpropagation;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Backpropagation for max pooling layers
 */
public class BackpropagationMaxPooling2D implements BackPropagationConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private BackPropagationConnectionCalculator cc;
    protected ValuesProvider activations;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (cc == null) {
	    cc = new BackpropagationMaxPooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider.getColumns());
	    cc.setActivations(activations);
	}

	cc.calculate(connections, valuesProvider, targetLayer);
    }

    @Override
    public float getLearningRate() {
	return cc.getLearningRate();
    }

    @Override
    public void setLearningRate(float learningRate) {
	cc.setLearningRate(learningRate);
    }

    @Override
    public float getMomentum() {
	return cc.getMomentum();
    }

    @Override
    public void setMomentum(float momentum) {
	cc.setMomentum(momentum);
    }

    @Override
    public float getL1weightDecay() {
	return cc.getL1weightDecay();
    }

    @Override
    public void setL1weightDecay(float weightDecay) {
	cc.setL1weightDecay(weightDecay);
    }

    @Override
    public float getL2weightDecay() {
	return cc.getL2weightDecay();
    }

    @Override
    public void setL2weightDecay(float l2weightDecay) {
	cc.setL2weightDecay(l2weightDecay);
    }
    @Override
    public ValuesProvider getActivations() {
	return cc != null ? cc.getActivations() : activations;
    }

    @Override
    public void setActivations(ValuesProvider activations) {
	this.activations = activations;
	if (cc != null) {
	    cc.setActivations(activations);
	}
    }

    public static class BackpropagationMaxPooling2DCC extends AparapiBackpropagationSubsampling2D {

	private static final long serialVersionUID = -8888670594631428090L;

	public BackpropagationMaxPooling2DCC(Subsampling2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected void pool(int inputStartIndex) {
	    int rl = regionLength;
	    int mbs = miniBatchSize;
	    int maxId = 0;
	    int ffActivationId = 0;
	    float max = 0;

	    for (int i = 0; i < mbs; i++) {
		ffActivationId = (inputStartIndex + featureMapOffsets[0]) * mbs + i;
		max = ffActivation[ffActivationId];
		for (int j = 1; j < rl; j++) {
		    ffActivationId = (inputStartIndex + featureMapOffsets[j]) * mbs + i;
		    float v = ffActivation[ffActivationId];
		    if (v > max) {
			maxId = ffActivationId;
			max = v;
		    }
		}

		input[maxId] = output[getGlobalId() * mbs + i];
	    }
	}
    }
}