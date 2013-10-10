package com.github.neuralnetworks.calculation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Util;

/**
 * 
 * this calculator calculates within the bounds of a single connections graph
 * 
 */
public class ConnectionGraphCalculator implements LayerCalculator {

    private final Connections connections;

    public ConnectionGraphCalculator(Connections connections) {
	super();
	this.connections = connections;
    }

    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	Util.fillArray(results.get(layer).getElements(), 0);

	Matrix result = results.get(layer);
	if (connections.getInputLayer() == layer) {
	    layer.getBackwardInputFunction().calculate(connections, results.get(connections.getOutputLayer()), result);
	} else if (connections.getOutputLayer() == layer) {
	    layer.getForwardInputFunction().calculate(connections, results.get(connections.getInputLayer()), result);
	}
    }
}
