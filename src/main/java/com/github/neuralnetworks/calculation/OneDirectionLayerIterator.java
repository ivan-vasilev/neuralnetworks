package com.github.neuralnetworks.calculation;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import com.github.neuralnetworks.architecture.Layer;

public abstract class OneDirectionLayerIterator implements LayerOrderStrategy {

	private Queue<Layer> layers;

	public OneDirectionLayerIterator(Layer layer) {
		super();

		layers = new LinkedList<>();
		setCurrentLayer(layer);
	}

	@Override
	public boolean hasNext() {
		return layers.size() > 0;
	}

	@Override
	public Layer next() {
		return layers.poll();
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
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
