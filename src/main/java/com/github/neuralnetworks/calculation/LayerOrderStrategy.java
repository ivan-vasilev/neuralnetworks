package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Layer;

/**
 *
 * the implementations of this interface provide a mechanism for retrieving the next layer from the propagation
 *
 */
public interface LayerOrderStrategy {
	public Layer getNextLayer();
	public void setCurrentLayer(Layer layer);
}
