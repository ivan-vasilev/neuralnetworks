package com.github.neuralnetworks.calculation;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;

public class ForwardLayerCalculator implements LayerCalculator {

	@Override
	public void calculate(Map<Layer, float[]> calculatedLayers, Layer layer, float[] result) {
		for (Connections c : layer.getInboundConnectionGraphs()) {
			layer.getInputFunction().calculateForward(c, calculatedLayers.get(c.getInputLayer()), result);
		}

		layer.getActivationFunction().value(result);
	}
}
