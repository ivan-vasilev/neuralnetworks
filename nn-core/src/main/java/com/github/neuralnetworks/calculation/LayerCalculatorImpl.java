package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.neuronfunctions.ConstantInput;
import com.github.neuralnetworks.neuronfunctions.InputFunction;
import com.github.neuralnetworks.util.Util;

public class LayerCalculatorImpl implements LayerCalculator {

    protected List<PropagationEventListener> listeners;

    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	calculate(calculatedLayers, new HashSet<Layer>(), results, layer);
    }

    protected void calculate(Set<Layer> calculatedLayers, Set<Layer> inProgressLayers, Map<Layer, Matrix> results, Layer currentLayer) {
	if (!calculatedLayers.contains(currentLayer) && !inProgressLayers.contains(currentLayer)) {
	    inProgressLayers.add(currentLayer);
	    for (Connections c : currentLayer.getConnectionGraphs()) {
		Layer opposite = c.getInputLayer() != currentLayer ? c.getInputLayer() : c.getOutputLayer();
		InputFunction inputFunction = c.getInputLayer() != currentLayer ? currentLayer.getForwardInputFunction() : currentLayer.getBackwardInputFunction();

		if (inputFunction instanceof ConstantInput) {
		    Matrix output = results.get(currentLayer);
		    if (output == null) {
			int columns = getInputColumns(results.values());
			output = new Matrix(currentLayer.getNeuronCount(), columns);
			results.put(currentLayer, output);
		    }

		    Util.fillArray(output.getElements(), ((ConstantInput) inputFunction).getOutput());
		    break;
		} else if (!inProgressLayers.contains(opposite)) {
		    calculate(calculatedLayers, inProgressLayers, results, opposite);
		    Matrix input = results.get(opposite);

		    Matrix output = results.get(currentLayer);
		    if (output == null) {
			output = new Matrix(currentLayer.getNeuronCount() * input.getColumns(), input.getColumns());
			results.put(currentLayer, output);
		    }

		    inputFunction.calculate(c, input, output);
		}
	    }

	    currentLayer.getActivationFunction().value(results.get(currentLayer));

	    inProgressLayers.remove(currentLayer);
	    calculatedLayers.add(currentLayer);

	    triggerEvent(new PropagationEvent(currentLayer, results));
	}
    }

    private Integer getInputColumns(Collection<Matrix> results) {
	for (Matrix m : results) {
	    if (m != null) {
		return m.getColumns();
	    }
	}

	return -1;
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
	if (listeners != null) {
	    for (PropagationEventListener l : listeners) {
		l.handleEvent(event);
	    }
	}
    }
}
