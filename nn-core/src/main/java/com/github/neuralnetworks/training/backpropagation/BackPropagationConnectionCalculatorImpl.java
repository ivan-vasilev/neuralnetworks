package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Connection calculator for the backpropagation phase of the algorithm
 * The difference with the regular ConnectionCalculatorImpl is that forwardBackprop's and backwardBackprop's properties (learing rate, momentum, weight decay) are updated before each propagation
 */
public abstract class BackPropagationConnectionCalculatorImpl implements ConnectionCalculator {

    private static final long serialVersionUID = -8854054073444883314L;

    private Properties properties;
    protected Map<Connections, BackpropagationConnectionCalculator> connectionCalculators;
    protected Set<BackpropagationConnectionCalculator> calculators;
    protected Layer currentLayer;
    protected int inputOutputSamples;

    public BackPropagationConnectionCalculatorImpl(Properties properties) {
	this.properties = properties;
	this.connectionCalculators = new HashMap<>();
	this.calculators = new HashSet<>();
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Integer> chunk = new TreeMap<>();
	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    if (!connectionCalculators.containsKey(e.getKey()) || targetLayer != currentLayer || inputOutputSamples != output.getColumns()) {
		chunk.put(e.getKey(), e.getValue().getElements().length);
	    }
	}

	if (chunk.size() > 0) {
	    addBackpropFunction(chunk, connectionCalculators, output.getColumns(), targetLayer);
	    calculators.addAll(connectionCalculators.values());
	    inputOutputSamples = output.getColumns();
	    currentLayer = targetLayer;
	}

	SortedMap<Connections, Matrix> chunkCalc = new TreeMap<>();
	for (BackpropagationConnectionCalculator bc : calculators) {
	    chunkCalc.clear();

	    Layer target = targetLayer;
	    Matrix out = output;
	    for (Entry<Connections, Matrix> e : connections.entrySet()) {
		if (connectionCalculators.get(e.getKey()) == bc) {
		    if (e.getKey().getInputLayer() instanceof BiasLayer) {
			chunkCalc.put(e.getKey(), output);
			target = e.getKey().getInputLayer();
			out = e.getValue();
		    } else {
			chunkCalc.put(e.getKey(), e.getValue());
		    }
		}
	    }

	    if (chunkCalc.size() > 0) {
		bc.setLearningRate(getLearningRate());
		bc.setMomentum(getMomentum());
		bc.setWeightDecay(getWeightDecay());
		bc.setActivations(getActivations());
		bc.calculate(chunkCalc, out, target);
	    }
	}
    }

    protected abstract void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, int inputOutputSamples, Layer targetLayer);

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
