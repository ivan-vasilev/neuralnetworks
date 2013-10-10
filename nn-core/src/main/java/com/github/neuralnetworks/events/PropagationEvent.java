package com.github.neuralnetworks.events;

import java.util.EventObject;
import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * This event is triggered when a propagation step is finished
 *
 * @author hok
 *
 */
public class PropagationEvent extends EventObject {

	private static final long serialVersionUID = 1L;

	private Map<Layer, Matrix> results;

	public PropagationEvent(Layer layer, Map<Layer, Matrix> results) {
		super(layer);
		this.results = results;
	}

	public Map<Layer, Matrix> getResults() {
		return results;
	}
}