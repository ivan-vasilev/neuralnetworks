package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.HashMap;
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
	protected List<PropagationEventListener> listeners;

	public Propagation() {
		super();
	}

	public Propagation(LayerCalculatorProvider layerCalculatorProvider, LayerOrderStrategy layerOrderStrategy) {
		super();
		this.calculated = new HashMap<Layer, float[]>();
		this.layerCalculatorProvider = layerCalculatorProvider;
		this.layerOrderStrategy = layerOrderStrategy;
	}

	/**
	 * @param values - values clamped to the input layer
	 * @param startLayer - input layer
	 * @return - the last calculated layer (presumably the output layer)
	 */
	public float[] propagate(float[] values, Layer startLayer) {
		float[] result = null;
		calculated.clear();
		calculated.put(startLayer, values);
		layerOrderStrategy.setCurrentLayer(startLayer);

		while (layerOrderStrategy.hasNext()) {
			Layer layer = layerOrderStrategy.next();
			result = new float[layer.getNeuronCount()];
			layerCalculatorProvider.getCalculator(layer).calculate(calculated, layer, result);
			calculated.put(layer, result);
			triggerEvent(new PropagationEvent(layer, calculated));
		}

		return result;
	}

	public void addEventListener(PropagationEventListener listener) {
		if (listeners == null) {
			listeners = new ArrayList<>();
		}

		listeners.add(listener);
	}

	public void removeEventListener(PropagationEventListener listener) {
		if (listeners != null) {
			listeners.remove(listener);
		}
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

	public LayerCalculatorProvider getLayerCalculatorProvider() {
		return layerCalculatorProvider;
	}

	public void setLayerCalculatorProvider(LayerCalculatorProvider layerCalculatorProvider) {
		this.layerCalculatorProvider = layerCalculatorProvider;
	}
}
