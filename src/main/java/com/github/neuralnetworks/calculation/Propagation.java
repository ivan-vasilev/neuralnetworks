package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.IConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;

/**
 *
 * implements propagation (forward and backward) of inputs through a network
 *
 */
public abstract class Propagation {

	protected Map<Layer, float[]> results;
	protected ICalculateLayer calculator;
	protected List<PropagationEventListener> listeners = new ArrayList<>();

	public Propagation(Map<Layer, float[]> results, ICalculateLayer calculator) {
		super();
		this.results = results;
		this.calculator = calculator;
	}

	protected List<Layer> getAdjacentOutputLayers(Layer layer) {
		List<Layer> result = new ArrayList<Layer>();
		if (layer.getOutboundConnectionGraphs() != null) {
			for (IConnections c : layer.getOutboundConnectionGraphs()) {
				result.add(c.getOutputLayer());
			}
		}

		return result;
	}

	protected List<Layer> getAdjacentInputLayers(Layer layer) {
		List<Layer> result = new ArrayList<Layer>();
		if (layer.getInboundConnectionGraphs() != null) {
			for (IConnections c : layer.getInboundConnectionGraphs()) {
				result.add(c.getOutputLayer());
			}
		}

		return result;
	}

	public void propagateForward() {
		while (hasMoreLayers()) {
			Layer layer = getNextLayer();
			results.put(layer, calculator.calculateForward(results, layer));
			triggerEvent(new PropagationEvent(layer, results));
		}
	}

	public void propagateBackward() {
		while (hasMoreLayers()) {
			Layer layer = getNextLayer();
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

	public Map<Layer, float[]> getResults() {
		return results;
	}

	public void setResults(Map<Layer, float[]> results) {
		this.results = results;
	}

	public void reset() {
		results.clear();
	}

	public abstract boolean hasMoreLayers();
	public abstract Layer getNextLayer();

	protected void triggerEvent(PropagationEvent event) {
		for (PropagationEventListener l : listeners) {
			l.handleEvent(event);
		}
	}
}
