package com.github.neuralnetworks.events;

import java.util.EventObject;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Triggered when a propagation step is finished
 */
public class PropagationEvent extends EventObject {

    private static final long serialVersionUID = 1L;

    private ValuesProvider results;

    public PropagationEvent(Layer layer, ValuesProvider results) {
	super(layer);
	this.results = results;
    }

    public ValuesProvider getResults() {
	return results;
    }
}