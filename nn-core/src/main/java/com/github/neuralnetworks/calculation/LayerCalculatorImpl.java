package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.neuroninput.ConstantInput;

public class LayerCalculatorImpl implements LayerCalculator {

	protected List<PropagationEventListener> listeners;

	@Override
	public void calculate(Map<Layer, float[]> calculated, Layer layer) {
		calculate(new HashSet<Layer>(), new HashSet<Connections>(), calculated, layer);
	}

	protected void calculate(Set<Layer> visitedLayers, Set<Connections> visitedConnections, Map<Layer, float[]> calculatedLayers, Layer currentLayer) {
		if (!visitedLayers.contains(currentLayer)) {
			visitedLayers.add(currentLayer);

			float[] layerResult = calculatedLayers.get(currentLayer);
			if (layerResult == null) {
				layerResult = new float[currentLayer.getNeuronCount()];
				calculatedLayers.put(currentLayer, layerResult);
			}

			if (currentLayer.getInputFunction() instanceof ConstantInput) {
				ConstantInput ci = (ConstantInput) currentLayer.getInputFunction();
				Arrays.fill(layerResult, ci.getOutput());
			} else {
				for (Connections c : currentLayer.getConnectionGraphs()) {
					if (!visitedConnections.contains(c)) {
						visitedConnections.add(c);
						calculateConnections(c, currentLayer, calculatedLayers);
					}
				}

				currentLayer.getActivationFunction().value(layerResult);

				triggerEvent(new PropagationEvent(currentLayer, calculatedLayers));
			}
		}
	}

	protected void calculateConnections(Connections c, Layer target, Map<Layer, float[]> calculatedLayers) {
		if (c.getInputLayer() != target) {
			target.getInputFunction().calculateForward(c, calculatedLayers.get(c.getInputLayer()), calculatedLayers.get(c.getOutputLayer()));
		} else {
			target.getInputFunction().calculateBackward(c, calculatedLayers.get(c.getOutputLayer()), calculatedLayers.get(c.getInputLayer()));
		}
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

	protected void triggerEvent(PropagationEvent event) {
		for (PropagationEventListener l : listeners) {
			l.handleEvent(event);
		}
	}
}
