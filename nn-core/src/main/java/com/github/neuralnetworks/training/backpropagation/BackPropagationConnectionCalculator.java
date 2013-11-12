package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class BackPropagationConnectionCalculator implements ConnectionCalculator {

    private static final long serialVersionUID = -8854054073444883314L;

    private Properties properties;
    private AparapiBackpropagationBaseByRows forwardBackprop;
    private AparapiBackpropagationBaseByColumns backwardBackprop;

    public BackPropagationConnectionCalculator(Properties properties, AparapiBackpropagationBaseByRows forwardBackprop, AparapiBackpropagationBaseByColumns backwardBackprop) {
	this.properties = properties;
	this.forwardBackprop = forwardBackprop;
	this.backwardBackprop = backwardBackprop;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Matrix> forward = new TreeMap<>();
	SortedMap<Connections, Matrix> backward = new TreeMap<>();
	SortedMap<Connections, Matrix> bias = new TreeMap<>();
	Matrix biasOutput = null;
	Layer biasLayer = null;
	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    if (c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		bias.put(c, output);
		biasOutput = input;
		biasLayer = c.getInputLayer();
	    } else if (c.getInputLayer() == targetLayer) {
		backward.put(c, input);
	    } else if (c.getOutputLayer() == targetLayer) {
		forward.put(c, input);
	    }
	}

	if (forward.size() > 0) {
	    forwardBackprop.setLearningRate(getLearningRate());
	    forwardBackprop.setMomentum(getMomentum());
	    forwardBackprop.setWeightDecay(getWeightDecay());
	    forwardBackprop.setActivations(getActivations());
	    forwardBackprop.calculate(forward, output, targetLayer);
	}

	if (backward.size() > 0) {
	    backwardBackprop.setLearningRate(getLearningRate());
	    backwardBackprop.setMomentum(getMomentum());
	    backwardBackprop.setWeightDecay(getWeightDecay());
	    backwardBackprop.setActivations(getActivations());
	    backwardBackprop.calculate(backward, output, targetLayer);
	}

	if (bias.size() > 0) {
	    backwardBackprop.setLearningRate(getLearningRate());
	    backwardBackprop.setMomentum(getMomentum());
	    backwardBackprop.setWeightDecay(0);
	    backwardBackprop.setActivations(getActivations());
	    backwardBackprop.calculate(bias, biasOutput, biasLayer);
	}
    }

    public float getLearningRate() {
	return properties.getParameter(Constants.LEARNING_RATE);
    }

    public float getMomentum() {
	return properties.getParameter(Constants.MOMENTUM);
    }

    public float getWeightDecay() {
	return properties.getParameter(Constants.WEIGHT_DECAY);
    }

    public Map<Layer, Matrix> getActivations() {
	return properties.getParameter(Constants.ACTIVATIONS);
    }
    
    public void setActivations(Map<Layer, Matrix> activations) {
	properties.setParameter(Constants.ACTIVATIONS, activations);
    }
}
