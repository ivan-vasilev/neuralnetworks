package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.IConnections;
import com.github.neuralnetworks.architecture.Neuron;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;

/**
 *
 * implements propagation (forward and backward) of inputs through a network
 *
 */
public abstract class Propagation {

	protected Map<Neuron[], float[]> results;
	protected ICalculateLayer calculator;
	protected List<PropagationEventListener> listeners = new ArrayList<>();

	public Propagation(Map<Neuron[], float[]> results, ICalculateLayer calculator) {
		super();
		this.results = results;
		this.calculator = calculator;
	}

	protected List<Neuron[]> getAdjacentOutputLayers(Neuron[] layer) {
		List<Neuron[]> result = new ArrayList<Neuron[]>();
		for (Neuron n : layer) {
			if (n.getOutboundConnections() != null) {
				for (IConnections c : n.getOutboundConnectionGraphs()) {
					result.add(c.getOutputNeurons());
				}
			}
		}

		return result;
	}

	protected List<Neuron[]> getAdjacentInputLayers(Neuron[] layer) {
		List<Neuron[]> result = new ArrayList<Neuron[]>();
		for (Neuron n : layer) {
			if (n.getInboundConnections() != null) {
				for (IConnections c : n.getInboundConnectionGraphs()) {
					result.add(c.getOutputNeurons());
				}
			}
		}

		return result;
	}

	public void propagateForward() {
		while (hasMoreLayers()) {
			Neuron[] layer = getNextLayer();
			results.put(layer, calculator.calculateForward(results, layer));
			triggerEvent(new PropagationEvent(layer, results));
		}
	}

	public void propagateBackward() {
		while (hasMoreLayers()) {
			Neuron[] layer = getNextLayer();
			results.put(layer, calculator.calculateBackward(results, layer));
			triggerEvent(new PropagationEvent(layer, results));
		}
	}

	public void addEventListener(PropagationEventListener listener) {
		listeners.add(listener);
	}

	public void removeEventListener(PropagationEventListener listener) {
		listeners.remove(listener);
	}

	public Map<Neuron[], float[]> getResults() {
		return results;
	}

	public void setResults(Map<Neuron[], float[]> results) {
		this.results = results;
	}

	public void reset() {
		results.clear();
	}

	public abstract boolean hasMoreLayers();
	public abstract Neuron[] getNextLayer();

	protected void triggerEvent(PropagationEvent event) {
		for (PropagationEventListener l : listeners) {
			l.handleEvent(event);
		}
	}
}
