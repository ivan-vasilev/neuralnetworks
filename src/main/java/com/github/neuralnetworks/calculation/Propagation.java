package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;

/**
 *
 * implements propagation (forward and backward) through a network
 *
 */
public class Propagation {

	protected Map<Layer, float[]> calculated;
	protected LayerCalculatorProvider layerCalculatorProvider;
	protected LayerOrderStrategy layerOrderStrategy;
	protected List<PropagationEventListener> listeners = new ArrayList<>();

	public Propagation(Map<Layer, float[]> calculated, LayerCalculatorProvider layerCalculatorProvider, LayerOrderStrategy layerOrderStrategy) {
		super();
		this.calculated = calculated;
		this.layerCalculatorProvider = layerCalculatorProvider;
		this.layerOrderStrategy = layerOrderStrategy;
	}

	public void propagate() {
		Layer layer = null; layerOrderStrategy.getNextLayer();
		while ((layer = layerOrderStrategy.getNextLayer()) != null) {
			calculated.put(layer, layerCalculatorProvider.getCalculator(layer).calculate(calculated, layer));
			triggerEvent(new PropagationEvent(layer, calculated));
		}
	}

	public void addEventListener(PropagationEventListener listener) {
		listeners.add(listener);
	}

	public void removeEventListener(PropagationEventListener listener) {
		listeners.remove(listener);
	}

	public Map<Layer, float[]> getCalculated() {
		return calculated;
	}

	public void setCalculated(Map<Layer, float[]> calculated) {
		this.calculated = calculated;
	}

	public void reset() {
		calculated.clear();
	}

	protected void triggerEvent(PropagationEvent event) {
		for (PropagationEventListener l : listeners) {
			l.handleEvent(event);
		}
	}

	public LayerOrderStrategy getLayerOrderStrategy() {
		return layerOrderStrategy;
	}

	public void setLayerOrderStrategy(LayerOrderStrategy layerOrderStrategy) {
		this.layerOrderStrategy = layerOrderStrategy;
	}
}
