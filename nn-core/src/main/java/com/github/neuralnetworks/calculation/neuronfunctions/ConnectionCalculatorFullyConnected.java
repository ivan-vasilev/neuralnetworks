package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Default implementation of Connection calculator for fully connected layers
 * Biases are also added
 * After all the input functions are calculated there is a list of activation functions that can be applied to the result
 * This class differs from LayerCalculatorImpl in the fact that LayerCalculatorImpl traverses the graph of layers, where ConnectionCalculatorImpl only deals with the connections passed as parameter
 * 
 * !!! Important !!!
 * The results of the calculations are represented as matrices (Matrix).
 * This is done, because it is assumed that implementations will provide a way for calculating many input results at once.
 * Each column of the matrix represents a single input. For example if the network is trained to classify MNIST images, each column of the input matrix will represent single MNIST image.
 */
public class ConnectionCalculatorFullyConnected implements ConnectionCalculator {

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
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	if (connections.size() > 0) {
	    SortedMap<Connections, Matrix> notBias = new TreeMap<>();
	    Set<GraphConnections> bias = new HashSet<>();

	    for (Entry<Connections, Matrix> e : connections.entrySet()) {
		Connections c = e.getKey();
		Matrix input = e.getValue();
		// bias layer scenarios
		if (Util.isBias(c.getInputLayer())) {
		    bias.add((GraphConnections) c);
		} else {
		    notBias.put(c, input);
		}
	    }

	    if (notBias.size() > 0) {
		if (preTransferFunctions != null && preTransferFunctions.size() > 0) {
		    for (MatrixFunction f : preTransferFunctions) {
			for (Entry<Connections, Matrix> e : connections.entrySet()) {
			    if (!Util.isBias(e.getKey().getInputLayer())) {
				f.value(e.getValue());
			    }
			}
		    }
		}

		calculateBias(bias, output);

		// new input function is required
		if (inputFunction == null || targetLayer != currentLayer || miniBatchSize != output.getColumns()) {
		    currentLayer = targetLayer;
		    miniBatchSize = output.getColumns();
		    SortedMap<GraphConnections, Integer> map = new TreeMap<>();
		    for (Entry<Connections, Matrix> e : notBias.entrySet()) {
			map.put((GraphConnections) e.getKey(), e.getValue().getElements().length);
		    }

		    inputFunction = createInputFunction(map, targetLayer);
		}

		inputFunction.calculate(notBias, output, targetLayer);

		if (activationFunctions != null) {
		    for (MatrixFunction f : activationFunctions) {
			f.value(output);
		    }
		}
	    }
	}
    }

    protected void calculateBias(Set<GraphConnections> bias, Matrix output) {
	if (bias.size() > 0) {
	    float[] out = output.getElements();
	    for (int i = 0; i < out.length; i++) {
		for (GraphConnections c : bias) {
		    out[i] += c.getConnectionGraph().getElements()[i / output.getColumns()];
		}
	    }
	}
    }


    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, Layer targetLayer) {
	return new AparapiWeightedSum(inputConnections, miniBatchSize, targetLayer);
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
