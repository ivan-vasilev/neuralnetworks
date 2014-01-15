package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.util.Util;

/**
 * Base class for implementations of the LayerCalculator interface
 */
public class LayerCalculatorBase implements Serializable {

    private static final long serialVersionUID = 1L;

    protected List<PropagationEventListener> listeners;
    protected Map<Layer, ConnectionCalculator> calculators;

    protected void calculate(Map<Layer, Matrix> results, List<ConnectionCalculateCandidate> connections) {
	if (connections.size() > 0) {
	    SortedMap<Connections, Matrix> map = new TreeMap<>();

	    for (int i = 0; i < connections.size(); i++) {
		ConnectionCalculateCandidate c = connections.get(i);
		map.put(c.connection, results.get(Util.getOppositeLayer(c.connection, c.target)));

		if (i == connections.size() - 1 || connections.get(i + 1).target != c.target) {
		    ConnectionCalculator cc = getConnectionCalculator(c.target);
		    if (cc != null) {
			Matrix output = getLayerResult(results, c.target);
			cc.calculate(map, output, c.target);
		    }

		    map.clear();

		    triggerEvent(new PropagationEvent(c.target, results));
		}
	    }
	}
    }

    protected Matrix getLayerResult(Map<Layer, Matrix> results, Layer layer) {
	Matrix result = results.get(layer);
	Integer columns = null;
	for (Entry<Layer, Matrix> e : results.entrySet()) {
	    if (columns == null || e.getValue().getColumns() < columns) {
		columns = e.getValue().getColumns();
	    }
	}

	if (result == null || result.getColumns() != columns) {
	    result = new Matrix(layer.getNeuronCount(), columns);
	    results.put(layer, result);
	} else {
	    Util.fillArray(result.getElements(), 0);
	}

	return result;
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

    public static class ConnectionCalculateCandidate {

	public Connections connection;
	public Layer target;

	public ConnectionCalculateCandidate(Connections connection, Layer target) {
	    super();
	    this.connection = connection;
	    this.target = target;
	}
    }
}
