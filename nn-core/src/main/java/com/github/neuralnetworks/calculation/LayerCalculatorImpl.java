package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
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

public class LayerCalculatorImpl implements LayerCalculator, Serializable {

    private static final long serialVersionUID = 1L;

    protected List<PropagationEventListener> listeners;

    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	calculate(calculatedLayers, new UniqueList<Layer>(), results, layer);
    }

    protected void calculate(Set<Layer> calculatedLayers, Set<Layer> inProgressLayers, Map<Layer, Matrix> results, Layer currentLayer) {
	if (!calculatedLayers.contains(currentLayer) && !inProgressLayers.contains(currentLayer)) {
	    inProgressLayers.add(currentLayer);
	    ConnectionCalculator cc = getConnectionCalculator(currentLayer);

	    if (cc != null) {
		SortedMap<Connections, Matrix> connections = new TreeMap<Connections, Matrix>();
		for (Connections c : currentLayer.getConnections()) {
		    Layer opposite = Util.getOppositeLayer(c, currentLayer);
		    if (!inProgressLayers.contains(opposite)) {
			calculate(calculatedLayers, inProgressLayers, results, opposite);
			connections.put(c, results.get(opposite));
		    }
		}

		Matrix output = results.get(currentLayer);
		int columns = getInputColumns(calculatedLayers, results);
		if (output == null || output.getColumns() != columns) {
		    output = new Matrix(currentLayer.getNeuronCount(), columns);
		    results.put(currentLayer, output);
		} else {
		    Util.fillArray(output.getElements(), 0);
		}

		cc.calculate(connections, output, currentLayer);
	    }

	    inProgressLayers.remove(currentLayer);
	    calculatedLayers.add(currentLayer);

	    triggerEvent(new PropagationEvent(currentLayer, results));
	}
    }

    protected ConnectionCalculator getConnectionCalculator(Layer layer){
	return layer.getConnectionCalculator();
    }

    private Integer getInputColumns(Set<Layer> calculatedLayers, Map<Layer, Matrix> results) {
	for (Entry<Layer, Matrix> e : results.entrySet()) {
	    if (calculatedLayers.contains(e.getKey())) {
		return e.getValue().getColumns();
	    }
	}

	return -1;
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
