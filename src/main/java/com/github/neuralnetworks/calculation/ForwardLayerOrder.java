package com.github.neuralnetworks.calculation;

import java.util.List;

import com.github.neuralnetworks.architecture.Layer;

public class ForwardLayerOrder extends OneDirectionLayerOrder {

	@Override
	protected List<Layer> getAdjacentLayers(Layer layer) {
		return layer.getAdjacentOutputLayers();
	}
}
