package com.github.neuralnetworks.calculation;

import java.util.Map;

import com.github.neuralnetworks.architecture.IConnections;
import com.github.neuralnetworks.architecture.Layer;

public class LayerCalculator implements ICalculateLayer {

	@Override
	public float[] calculateForward(Map<Layer, float[]> calculatedLayers, Layer layer) {
		float[] inputValues = new float[layer.getNeuronCount()];
		for (IConnections c : layer.getInboundConnectionGraphs()) {
			layer.getInputFunction().calculateForward(c, calculatedLayers.get(c.getInputLayer()), inputValues);
		}

		return layer.getActivationFunction().value(inputValues);
	}

	@Override
	public float[] calculateBackward(Map<Layer, float[]> calculatedLayers, Layer layer) {
		float[] inputValues = new float[layer.getNeuronCount()];
		for (IConnections c : layer.getOutboundConnectionGraphs()) {
			layer.getInputFunction().calculateBackward(c, calculatedLayers.get(c.getOutputLayer()), inputValues);
		}

		return layer.getActivationFunction().value(inputValues);
	}
}
