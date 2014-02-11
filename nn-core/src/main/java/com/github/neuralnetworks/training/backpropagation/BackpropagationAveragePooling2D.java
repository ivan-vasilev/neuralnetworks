package com.github.neuralnetworks.training.backpropagation;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Backpropagation for average pooling layers
 */
public class BackpropagationAveragePooling2D implements BackpropagationConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private BackpropagationConnectionCalculator cc;
    protected ValuesProvider activations;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (cc == null) {
	    cc = new BackpropagationAveragePooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider.getColumns());
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
    public float getWeightDecay() {
	return cc.getWeightDecay();
    }

    @Override
    public void setWeightDecay(float weightDecay) {
	cc.setWeightDecay(weightDecay);
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

    public class BackpropagationAveragePooling2DCC extends AparapiBackpropagationSubsampling2D {

	private static final long serialVersionUID = -8888670594631428090L;

	public BackpropagationAveragePooling2DCC(Subsampling2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected void pool(int inputStartIndex) {
	    int rl = regionLength;
	    int miniBatch = miniBatchSize;
	    float div = 0;

	    for (int i = 0; i < miniBatch; i++) {
		div = output[getGlobalId() * miniBatch + i] / rl;
		for (int j = 0; j < rl; j++) {
		    input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i] = div;
		}
	    }
	}
    }
}
