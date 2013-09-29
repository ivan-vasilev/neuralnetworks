package com.github.neuralnetworks.calculation;

import java.util.List;

import com.github.neuralnetworks.architecture.Layer;

public class BackwardLayerIterator extends OneDirectionLayerIterator {

	public BackwardLayerIterator(Layer layer) {
		super(layer);
	}

	@Override
	protected List<Layer> getAdjacentLayers(Layer layer) {
		return layer.getAdjacentInputLayers();
	}
}
