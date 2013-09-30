package com.github.neuralnetworks.calculation;

import java.util.Iterator;

import com.github.neuralnetworks.architecture.Layer;

/**
 *
 * the implementations of this interface provide a mechanism for retrieving the next layer from the propagation
 *
 */
public interface LayerOrderStrategy extends Iterator<Layer> {
	public void setCurrentLayer(Layer layer);
}
