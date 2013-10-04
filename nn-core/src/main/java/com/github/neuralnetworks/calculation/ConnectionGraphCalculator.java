package com.github.neuralnetworks.calculation;

import java.util.Arrays;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;

/**
 *
 * this calculator calculates within the bounds of a single connections graph
 *
 */
public class ConnectionGraphCalculator implements LayerCalculator {

	private Connections connections;

	public ConnectionGraphCalculator(Connections connections) {
		super();
		this.connections = connections;
	}

	@Override
	public void calculate(Map<Layer, float[]> calculatedLayers, Layer layer) {
		if (!calculatedLayers.containsKey(layer)) {
			calculatedLayers.put(layer, new float[layer.getNeuronCount()]);
		} else {
			Arrays.fill(calculatedLayers.get(layer), 0);
		}

		float[] result = calculatedLayers.get(layer);
		if (connections.getInputLayer() == layer) {
			layer.getInputFunction().calculateBackward(connections, calculatedLayers.get(connections.getOutputLayer()), result);
		} else if (connections.getOutputLayer() == layer) {
			layer.getInputFunction().calculateForward(connections, calculatedLayers.get(connections.getInputLayer()), result);
		}
	}
}
