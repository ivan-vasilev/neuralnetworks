package com.github.neuralnetworks.training.backpropagation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Connection calculator for the backpropagation phase of the algorithm
 * The difference with the regular ConnectionCalculatorImpl is that forwardBackprop's and backwardBackprop's properties (learing rate, momentum, weight decay) are updated before each propagation
 */
public abstract class BackPropagationConnectionCalculatorImpl implements ConnectionCalculator {

    private static final long serialVersionUID = -8854054073444883314L;

    private Properties properties;
    protected Map<Connections, BackpropagationConnectionCalculator> connectionCalculators;
    protected Set<BackpropagationConnectionCalculator> calculators;
    protected ValuesProvider activations;
    protected Layer currentLayer;
    protected int miniBatchSize;

    public BackPropagationConnectionCalculatorImpl(Properties properties) {
	this.properties = properties;
	this.connectionCalculators = new HashMap<>();
	this.calculators = new HashSet<>();
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	SortedMap<Connections, Integer> chunk = new TreeMap<>();
	for (Connections c : connections) {
	    if (!connectionCalculators.containsKey(c) || targetLayer != currentLayer || miniBatchSize != valuesProvider.getColumns()) {
		chunk.put(c, valuesProvider.getColumns() * valuesProvider.getUnitCount(Util.getOppositeLayer(c, targetLayer), c));
	    }
	}

	if (chunk.size() > 0) {
	    miniBatchSize = valuesProvider.getColumns();
	    currentLayer = targetLayer;
	    addBackpropFunction(chunk, connectionCalculators, targetLayer);
	    calculators.addAll(connectionCalculators.values());
	}

	List<Connections> chunkCalc = new ArrayList<>();
	for (BackpropagationConnectionCalculator bc : calculators) {
	    chunkCalc.clear();

	    Layer target = targetLayer;
	    for (Connections c : connections) {
		if (connectionCalculators.get(c) == bc) {
		    chunkCalc.add(c);
		    if (Util.isBias(c.getInputLayer()) && c.getInputLayer() != targetLayer) {
			target = c.getInputLayer();
		    }
		}
	    }

	    if (chunkCalc.size() > 0) {
		bc.setLearningRate(getLearningRate());
		bc.setMomentum(getMomentum());
		bc.setWeightDecay(getWeightDecay());
		bc.setActivations(getActivations());
		bc.calculate(chunkCalc, valuesProvider, target);
	    }
	}
    }

    protected abstract void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, Layer targetLayer);

    public float getLearningRate() {
	return properties.getParameter(Constants.LEARNING_RATE);
    }

    public int getMiniBatchSize() {
	return miniBatchSize;
    }

    public float getMomentum() {
	return properties.getParameter(Constants.MOMENTUM);
    }

    public float getWeightDecay() {
	return properties.getParameter(Constants.WEIGHT_DECAY);
    }

    public ValuesProvider getActivations() {
	return activations;
    }
    
    public void setActivations(ValuesProvider activations) {
	this.activations = activations;
    }
}
