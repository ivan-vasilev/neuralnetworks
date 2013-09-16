package com.github.neuralnetworks.calculation;

import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import com.github.neuralnetworks.architecture.Layer;

/**
 * one direction propagation (forward or backward)
 *
 */
public class OneDirectionPropagation extends Propagation {

	private Queue<Layer> layers = new LinkedList<>();

	public OneDirectionPropagation(Map<Layer, float[]> results, ICalculateLayer calculator) {
		super(results, calculator);
	}

	@Override
	public boolean hasMoreLayers() {
		return layers.size() > 0;
	}

	@Override
	public Layer getNextLayer() {
		return layers.poll();
	}

	@Override
	public void propagateForward() {
		Queue<Layer> queue = new LinkedList<>();
		queue.addAll(results.keySet());

		while (queue.size() > 0) {
			layers.addAll(getAdjacentOutputLayers(queue.poll()));
		}

		super.propagateForward();
	}

	@Override
	public void propagateBackward() {
		Queue<Layer> queue = new LinkedList<>();
		queue.addAll(results.keySet());

		while (queue.size() > 0) {
			queue.addAll(getAdjacentInputLayers(queue.poll()));
		}

		super.propagateForward();
	}

	@Override
	public void reset() {
		super.reset();
		layers.clear();
	}
}
