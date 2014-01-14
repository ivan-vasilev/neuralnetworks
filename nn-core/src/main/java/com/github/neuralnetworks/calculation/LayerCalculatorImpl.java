package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Default Implementation of the LayerCalculator interface
 * It takes advantage of the fact that the neural network is a graph with layers as nodes and connections between layers as links of the graph
 * The results are propagated within the graph
 */
public class LayerCalculatorImpl implements LayerCalculator, Serializable {

    private static final long serialVersionUID = 1L;

    protected List<PropagationEventListener> listeners;
    protected Map<Layer, ConnectionCalculator> calculators;

    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	calculate(calculatedLayers, new UniqueList<Layer>(), results, layer);
    }

    /**
     * Calculates single layer based on the network graph
     * 
     * For example the feedforward part of the backpropagation algorithm the initial parameters would be:
     * "currentLayer" will be the output layer of the network
     * "results" will contain only one entry for the input layer of the network - this is the training example
     * "calculatedLayers" will contain only one entry - the input layer
     * "inProgressLayers" will be empty
     * 
     * In the backpropagation part the initial parameters would be:
     * "currentLayer" will be the input layer of the network
     * "results" will contain only one entry for the output layer of the network - this is the calculated error derivative between the result of the network and the target value
     * "calculatedLayers" will contain only one entry - the output layer
     * "inProgressLayers" will be empty
     * 
     * This allows for single code to be used for the whole backpropagation, but also for RBMs, autoencoders, etc
     * 
     * @param calculatedLayers - layers that are fully calculated - the results for these layers can be used for calculating other parts of the network
     * @param inProgressLayers - layers which are currently calculated, but are not yet finished - not all connections to the layer are calculated and the result of the propagation through this layer cannot be used for another calculations
     * @param results - results of the calculations
     * @param currentLayer - the layer which is currently being calculated.
     * @return
     */
    protected boolean calculate(Set<Layer> calculatedLayers, Set<Layer> inProgressLayers, Map<Layer, Matrix> results, Layer currentLayer) {
	boolean result = false;

	if (calculatedLayers.contains(currentLayer)) {
	    result = true;
	} else if (!inProgressLayers.contains(currentLayer)) {
	    inProgressLayers.add(currentLayer);
	    ConnectionCalculator cc = getConnectionCalculator(currentLayer);

	    if (cc != null) {
		SortedMap<Connections, Matrix> connections = new TreeMap<Connections, Matrix>();
		for (Connections c : currentLayer.getConnections()) {
		    Layer opposite = Util.getOppositeLayer(c, currentLayer);
		    if (calculate(calculatedLayers, inProgressLayers, results, opposite)) {
			connections.put(c, results.get(opposite));
		    }
		}

		Matrix output = getLayerResult(calculatedLayers, results, currentLayer);
		cc.calculate(connections, output, currentLayer);
	    }

	    result = true;

	    inProgressLayers.remove(currentLayer);
	    calculatedLayers.add(currentLayer);

	    triggerEvent(new PropagationEvent(currentLayer, results));
	}

	return result;
    }

    protected Matrix getLayerResult(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	Matrix result = results.get(layer);
	int columns = getInputColumns(calculatedLayers, results);
	if (result == null || result.getColumns() != columns) {
	    result = new Matrix(layer.getNeuronCount(), columns);
	    results.put(layer, result);
	} else {
	    Util.fillArray(result.getElements(), 0);
	}

	return result;
    }

    protected Integer getInputColumns(Set<Layer> calculatedLayers, Map<Layer, Matrix> results) {
	for (Entry<Layer, Matrix> e : results.entrySet()) {
	    if (calculatedLayers.contains(e.getKey())) {
		return e.getValue().getColumns();
	    }
	}

	return -1;
    }

    public void addConnectionCalculator(Layer layer, ConnectionCalculator calculator) {
	if (calculators == null) {
	    calculators = new HashMap<>();
	}

	calculators.put(layer, calculator);
    }

    public ConnectionCalculator getConnectionCalculator(Layer layer) {
	if (calculators != null) {
	    return calculators.get(layer);
	}

	return null;
    }

    public void removeConnectionCalculator(Layer layer) {
	if (calculators != null) {
	    calculators.remove(layer);
	}
    }

    public void addEventListener(PropagationEventListener listener) {
	if (listeners == null) {
	    listeners = new ArrayList<>();
	}

	listeners.add(listener);
    }

    public void removeEventListener(PropagationEventListener listener) {
	if (listeners != null) {
	    listeners.remove(listener);
	}
    }

    protected void triggerEvent(PropagationEvent event) {
	if (listeners != null) {
	    for (PropagationEventListener l : listeners) {
		l.handleEvent(event);
	    }
	}
    }
}
