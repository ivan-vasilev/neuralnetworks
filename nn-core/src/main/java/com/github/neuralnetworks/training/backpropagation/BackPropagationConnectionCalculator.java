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

/**
 * Connection calculator for the backpropagation phase of the algorithm
 * The difference with the regular ConnectionCalculatorImpl is that forwardBackprop's and backwardBackprop's properties (learing rate, momentum, weight decay) are updated before each propagation
 */
public class BackPropagationConnectionCalculator implements ConnectionCalculator {

    private static final long serialVersionUID = -8854054073444883314L;

    private Properties properties;
    private AparapiBackpropagationBase backprop;

    public BackPropagationConnectionCalculator(Properties properties, AparapiBackpropagationBase backprop) {
	this.properties = properties;
	this.backprop = backprop;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Matrix> noBias = new TreeMap<>();
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
	    } else {
		noBias.put(c, input);
	    }
	}

	if (noBias.size() > 0) {
	    backprop.setLearningRate(getLearningRate());
	    backprop.setMomentum(getMomentum());
	    backprop.setWeightDecay(getWeightDecay());
	    backprop.setActivations(getActivations());
	    backprop.calculate(noBias, output, targetLayer);
	}

	if (bias.size() > 0) {
	    backprop.setLearningRate(getLearningRate());
	    backprop.setMomentum(getMomentum());
	    backprop.setWeightDecay(0);
	    backprop.setActivations(getActivations());
	    backprop.calculate(bias, biasOutput, biasLayer);
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
