package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Default implementation of Connection calculator for fully connected layers
 * Biases are also added After all the input functions are calculated there is a
 * list of activation functions that can be applied to the result This class
 * differs from LayerCalculatorImpl in the fact that LayerCalculatorImpl
 * traverses the graph of layers, where ConnectionCalculatorImpl only deals with
 * the connections passed as parameter
 * 
 * !!! Important !!! The results of the calculations are represented as matrices
 * (Matrix). This is done, because it is assumed that implementations will
 * provide a way for calculating many input results at once. Each column of the
 * matrix represents a single input. For example if the network is trained to
 * classify MNIST images, each column of the input matrix will represent single
 * MNIST image.
 */
public class ConnectionCalculatorFullyConnected implements ConnectionCalculator, PropagationEventListener {

    private static final long serialVersionUID = -5405654469496055017L;

    protected ConnectionCalculator inputFunction;
    protected Layer currentLayer;
    protected int miniBatchSize;

    /**
     * Activation functions that are executed before the transfer function
     */
    protected List<MatrixFunction> preTransferFunctions;

    /**
     * Activation functions that are called after the transfer function
     */
    protected List<MatrixFunction> activationFunctions;

    public ConnectionCalculatorFullyConnected() {
	super();
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (connections.size() > 0) {
	    List<Connections> notBias = new ArrayList<>();
	    Connections bias = null;

	    for (Connections c : connections) {
		// bias layer scenarios
		if (Util.isBias(c.getInputLayer())) {
		    bias = c;
		} else {
		    notBias.add(c);
		}
	    }

	    if (notBias.size() > 0) {
		if (preTransferFunctions != null && preTransferFunctions.size() > 0) {
		    preTransferFunctions.forEach(f -> connections.stream().filter(c -> !Util.isBias(c.getInputLayer())).forEach(c -> f.value(valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c))));
		}

		calculateBias(bias, valuesProvider);

		// new input function is required
		if (inputFunction == null || targetLayer != currentLayer || miniBatchSize != valuesProvider.getColumns()) {
		    miniBatchSize = valuesProvider.getColumns();
		    currentLayer = targetLayer;
		    SortedMap<GraphConnections, Integer> map = new TreeMap<>();
		    notBias.forEach(c -> map.put((GraphConnections) c, valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c).getElements().length));
		    inputFunction = createInputFunction(map, valuesProvider, targetLayer);
		}

		inputFunction.calculate(notBias, valuesProvider, targetLayer);

		if (activationFunctions != null) {
		    activationFunctions.forEach(f -> f.value(valuesProvider.getValues(targetLayer, notBias)));
		}
	    }
	}
    }

    protected void calculateBias(Connections bias, ValuesProvider valuesProvider) {
	if (bias != null) {
	    float[] biasValue = valuesProvider.getValues(bias.getInputLayer(), bias).getElements();
	    if (biasValue[0] == 0) {
		Util.fillArray(biasValue, 1);
	    }

	    float[] out = valuesProvider.getValues(bias.getOutputLayer(), bias).getElements();
	    for (int i = 0; i < out.length; i++) {
		GraphConnections gc = (GraphConnections) bias;
		out[i] += gc.getConnectionGraph().getElements()[i / valuesProvider.getColumns()];
	    }
	}
    }

    @Override
    public void handleEvent(PropagationEvent event) {
	if (preTransferFunctions != null) {
	    preTransferFunctions.stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
	}

	if (activationFunctions != null) {
	    activationFunctions.stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
	}
    }

    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiWeightedSum(inputConnections, valuesProvider.getColumns(), targetLayer);
    }

    public void addPreTransferFunction(MatrixFunction function) {
	if (preTransferFunctions == null) {
	    preTransferFunctions = new UniqueList<>();
	}

	preTransferFunctions.add(function);
    }

    public void removePreTransfer(MatrixFunction function) {
	if (preTransferFunctions != null) {
	    preTransferFunctions.remove(function);
	}
    }

    public void addActivationFunction(MatrixFunction activationFunction) {
	if (activationFunctions == null) {
	    activationFunctions = new UniqueList<>();
	}

	activationFunctions.add(activationFunction);
    }

    public void removeActivationFunction(MatrixFunction activationFunction) {
	if (activationFunctions != null) {
	    activationFunctions.remove(activationFunction);
	}
    }

    public ConnectionCalculator getInputFunction() {
	return inputFunction;
    }

    public void setInputFunction(ConnectionCalculator inputFunction) {
	this.inputFunction = inputFunction;
    }
}
