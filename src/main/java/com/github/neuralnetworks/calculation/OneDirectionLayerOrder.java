package com.github.neuralnetworks.calculation;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import com.github.neuralnetworks.architecture.Layer;

public abstract class OneDirectionLayerOrder implements LayerOrderStrategy {

	private Queue<Layer> layers = new LinkedList<>();

	public OneDirectionLayerOrder() {
		super();
	}

	public OneDirectionLayerOrder(Layer currentLayer) {
		super();
		setCurrentLayer(currentLayer);
	}

	@Override
	public Layer getNextLayer() {
		return layers.poll();
	}

	@Override
	public void setCurrentLayer(Layer layer) {
		layers.clear();

		Queue<Layer> queue = new LinkedList<>();
		queue.add(layer);

		while (queue.size() > 0) {
			List<Layer> adjacent = getAdjacentLayers(layer);
			queue.addAll(adjacent);
			layers.addAll(adjacent);
		}
	}

	protected abstract List<Layer> getAdjacentLayers(Layer layer);
}
