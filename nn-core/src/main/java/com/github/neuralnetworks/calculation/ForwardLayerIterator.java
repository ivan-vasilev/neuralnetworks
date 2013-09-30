package com.github.neuralnetworks.calculation;

import java.util.List;

import com.github.neuralnetworks.architecture.Layer;

public class ForwardLayerIterator extends OneDirectionLayerIterator {

	public ForwardLayerIterator(Layer layer) {
		super(layer);
	}

	@Override
	protected List<Layer> getAdjacentLayers(Layer layer) {
		return layer.getAdjacentOutputLayers();
	}
}
