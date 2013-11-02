package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumByColumns;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumByRows;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Aparapi implementation of the backpropagation algorithm
 */
public class BackPropagationAparapiImpl extends LayerCalculatorImpl implements BackPropagation, ConnectionCalculator {

    private static final long serialVersionUID = 1L;

    private Map<Layer, Matrix> results;
    private AparapiRmsDerivativeByRows forwardBackprop;
    private AparapiRmsDerivativeByColumns backwardBackprop;

    public BackPropagationAparapiImpl(float learningRate, float momentum) {
	super();
	this.results = new HashMap<Layer, Matrix>();
	forwardBackprop = new AparapiRmsDerivativeByRows();
	backwardBackprop = new AparapiRmsDerivativeByColumns();
	forwardBackprop.learningRate = backwardBackprop.learningRate = learningRate;
	forwardBackprop.momentum = backwardBackprop.momentum = momentum;
    }

    @Override
    public void backPropagate(Map<Layer, Matrix> activations, Matrix outputError, NeuralNetwork layers) {
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(layers.getOutputLayer());
	for (Matrix m : results.values()) {
	    Util.fillArray(m.getElements(), 0);
	}

	results.put(layers.getOutputLayer(), outputError);
	forwardBackprop.activations = backwardBackprop.activations = activations;

	calculate(calculatedLayers, results, layers.getInputLayer());
    }

    @Override
    protected ConnectionCalculator getConnectionCalculator(Layer layer) {
	return this;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	SortedMap<Connections, Matrix> forward = new TreeMap<>();
	SortedMap<Connections, Matrix> backward = new TreeMap<>();
	SortedMap<Connections, Matrix> bias = new TreeMap<>();
	Matrix biasOutput = null;
	Layer biasLayer = null;
	for (Entry<Connections, Matrix> e : connections.entrySet()) {
	    Connections c = e.getKey();
	    Matrix input = e.getValue();
	    if (c.getInputLayer().getConnectionCalculator() instanceof ConstantConnectionCalculator) {
		bias.put(c, output);
		biasOutput = input;
		biasLayer = c.getInputLayer();
	    } else if (c.getInputLayer() == targetLayer) {
		backward.put(c, input);
	    } else if (c.getOutputLayer() == targetLayer) {
		forward.put(c, input);
	    }
	}

	if (forward.size() > 0) {
	    forwardBackprop.calculate(forward, output, targetLayer);
	}

	if (backward.size() > 0) {
	    backwardBackprop.calculate(backward, output, targetLayer);
	}

	if (bias.size() > 0) {
	    backwardBackprop.calculate(bias, biasOutput, biasLayer);
	}
    }

    @Override
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target) {
	if (activation.getElements().length != target.getElements().length || activation.getColumns() != target.getColumns()) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	Matrix result = new Matrix(activation);
	for (int i = 0; i < activation.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - activation.getElements()[i]) * activation.getElements()[i] * (1 - activation.getElements()[i]);
	}

	return result;
    }

    private static class AparapiRmsDerivativeByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] outputActivation;
	private float[] weightUpdates;
	private Map<Layer, float[]> storedWeightUpdates = new HashMap<>();
	private float learningRate;
	private float momentum;
	private Map<Layer, Matrix> activations;

	@Override
	public void calculate(SortedMap<Connections, Matrix> inputConnections, Matrix outputMatrix, Layer targetLayer) {
	    super.calculate(inputConnections, outputMatrix, targetLayer);
	    if (inputConnections.size() > 1) {
		int i = 0;
		for (java.util.Map.Entry<Connections, Matrix> e : inputConnections.entrySet()) {
		    System.arraycopy(input, inputStartPositions[i], e.getValue().getElements(), 0, e.getValue().getElements().length);
		    System.arraycopy(weights, weightStartPositions[i], e.getKey().getConnectionGraph().getElements(), 0, e.getKey().getConnectionGraph().getElements().length);
		    i++;
		}
	    }
	}

	@Override
	protected void init(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);

	    weightUpdates = storedWeightUpdates.get(targetLayer);
	    if (weightUpdates == null) {
		weightUpdates = new float[weights.length];
		storedWeightUpdates.put(targetLayer, weightUpdates);
	    }

	    outputActivation = activations.get(targetLayer).getElements();
	}

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);

	    for (int i = 0; i < series; i++) {
		for (int j = 0; j < weightsColumns[i]; j++) {
		    int weightIndex = weightIndex(row, j, i);
		    float weightUpdate = learningRate * input[inputIndex(j, column, i)] * outputActivation[outputIndex(row, column, i)] + momentum * weightUpdates[weightIndex];
		    weights[weightIndex] += weightUpdate;
		    weightUpdates[weightIndex] = weightUpdate;
		}
	    }
	}
    }

    private static class AparapiRmsDerivativeByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] outputActivation;
	private Map<Layer, float[]> storedWeightUpdates = new HashMap<>();
	private float[] weightUpdates;
	private float learningRate;
	private float momentum;
	private Map<Layer, Matrix> activations;

	@Override
	public void calculate(SortedMap<Connections, Matrix> inputConnections, Matrix outputMatrix, Layer targetLayer) {
	    super.calculate(inputConnections, outputMatrix, targetLayer);
	    if (inputConnections.size() > 1) {
		int i = 0;
		for (java.util.Map.Entry<Connections, Matrix> e : inputConnections.entrySet()) {
		    System.arraycopy(input, inputStartPositions[i], e.getValue().getElements(), 0, e.getValue().getElements().length);
		    System.arraycopy(weights, weightStartPositions[i], e.getKey().getConnectionGraph().getElements(), 0, e.getKey().getConnectionGraph().getElements().length);
		    i++;
		}
	    }
	}

	@Override
	protected void init(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);

	    weightUpdates = storedWeightUpdates.get(targetLayer);
	    if (weightUpdates == null) {
		weightUpdates = new float[weights.length];
		storedWeightUpdates.put(targetLayer, weightUpdates);
	    }

	    outputActivation = activations.get(targetLayer).getElements();
	}

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);

	    for (int i = 0; i < series; i++) {
		for (int j = 0; j < weightsRows[i]; j++) {
		    int weightIndex = weightIndex(j, row, i);
		    float weightUpdate = learningRate * input[inputIndex(j, column, i)] * outputActivation[outputIndex(row, column, i)] + momentum * weightUpdates[weightIndex];
		    weights[weightIndex] += weightUpdate;
		    weightUpdates[weightIndex] = weightUpdate;
		}
	    }
	}
    }
}