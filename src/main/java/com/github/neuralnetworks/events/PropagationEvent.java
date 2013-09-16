package com.github.neuralnetworks.events;

import java.util.EventObject;
import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;

/**
 * This event is triggered when a propagation step is finished
 *
 * @author hok
 *
 */
public class PropagationEvent extends EventObject {

	private static final long serialVersionUID = 1L;

	private Map<Layer, float[]> results;

	public PropagationEvent(Layer layer, Map<Layer, float[]> results) {
		super(layer);
		this.results = results;
	}

	public Map<Layer, float[]> getResults() {
		return results;
	}

}