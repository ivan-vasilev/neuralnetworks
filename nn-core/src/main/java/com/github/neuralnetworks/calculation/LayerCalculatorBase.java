package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.util.Tensor;

/**
 * Base class for implementations of the LayerCalculator interface
 */
public class LayerCalculatorBase implements Serializable {

    private static final long serialVersionUID = 1L;

    protected List<PropagationEventListener> listeners;
    protected Map<Layer, ConnectionCalculator> calculators = new HashMap<>();

    protected void calculate(ValuesProvider valuesProvider, List<ConnectionCandidate> connections, NeuralNetwork nn) {
	if (connections.size() > 0) {
	    List<Connections> chunk = new ArrayList<>();

	    for (int i = 0; i < connections.size(); i++) {
		ConnectionCandidate c = connections.get(i);
		chunk.add(c.connection);

		if (i == connections.size() - 1 || connections.get(i + 1).target != c.target) {
		    ConnectionCalculator cc = getConnectionCalculator(c.target);
		    if (cc != null) {
			Tensor t = valuesProvider.getValues(c.target, chunk);
			t.forEach(j -> t.getElements()[j] = 0);
			cc.calculate(chunk, valuesProvider, c.target);
		    }

		    chunk.clear();

		    triggerEvent(new PropagationEvent(c.target, chunk, nn, valuesProvider));
		}
	    }
	}
    }

    public void addConnectionCalculator(Layer layer, ConnectionCalculator calculator) {
	calculators.put(layer, calculator);
    }

    public ConnectionCalculator getConnectionCalculator(Layer layer) {
	return calculators.get(layer);
    }

    public void removeConnectionCalculator(Layer layer) {
	calculators.remove(layer);
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
	    listeners.forEach(l -> l.handleEvent(event));
	}

	if (calculators != null) {
	    calculators.values().stream().filter(cc -> cc instanceof PropagationEventListener).forEach(cc -> ((PropagationEventListener) cc).handleEvent(event));
	}
    }
}
