package com.github.neuralnetworks.calculation;

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
	public void calculate(Map<Layer, float[]> calculatedLayers, Layer layer, float[] result) {
		if (connections.getInputLayer() == layer) {
			result = new float[layer.getNeuronCount()];
			layer.getInputFunction().calculateBackward(connections, calculatedLayers.get(connections.getOutputLayer()), result);
		} else if (connections.getOutputLayer() == layer) {
			result = new float[layer.getNeuronCount()];
			layer.getInputFunction().calculateForward(connections, calculatedLayers.get(connections.getInputLayer()), result);
		}
	}

}
