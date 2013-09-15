package com.github.neuralnetworks.events;

import java.util.EventObject;
import java.util.Map;

import com.github.neuralnetworks.architecture.Neuron;

/**
 * This event is triggered when a propagation step is finished
 *
 * @author hok
 *
 */
public class PropagationEvent extends EventObject {

	private static final long serialVersionUID = 1L;

	private Map<Neuron[], float[]> results;

	public PropagationEvent(Neuron[] layer, Map<Neuron[], float[]> results) {
		super(layer);
		this.results = results;
	}

	public Map<Neuron[], float[]> getResults() {
		return results;
	}

}