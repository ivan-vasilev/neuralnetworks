package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.InputOutputLayers;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.neuronfunctions.AparapiWeightedSumByColumns;
import com.github.neuralnetworks.neuronfunctions.AparapiWeightedSumByRows;
import com.github.neuralnetworks.neuronfunctions.InputFunction;
import com.github.neuralnetworks.util.Util;

/**
 * Aparapi implementation of the backpropagation algorithm
 * 
 */
public class BackPropagationAparapiImpl extends LayerCalculatorImpl implements BackPropagation {

    private static final long serialVersionUID = 1L;

    private Map<Layer, Matrix> results;
    private Map<Layer, Matrix> activations;
    private AparapiRmsDerivativeByRows forward;
    private AparapiRmsDerivativeByColumns backward;

    public BackPropagationAparapiImpl(float learningRate, float momentum) {
	super();
	this.results = new HashMap<Layer, Matrix>();
	forward = new AparapiRmsDerivativeByRows();
	backward = new AparapiRmsDerivativeByColumns();
	forward.learningRate = backward.learningRate = learningRate;
	forward.momentum = backward.momentum = momentum;
    }

    @Override
    protected InputFunction getInputFunction(Connections connection, Layer layer) {
	if (layer != connection.getInputLayer()) {
	    forward.activation = activations.get(layer).getElements();
	    return forward;
	} else {
	    backward.activation = activations.get(layer).getElements();
	    return backward;
	}
    }

    @Override
    public void backPropagate(Map<Layer, Matrix> activations, Matrix outputError, InputOutputLayers layers) {
	Set<Layer> calculatedLayers = new HashSet<Layer>();
	calculatedLayers.add(layers.getOutputLayer());
	for (Matrix m : results.values()) {
	    Util.fillArray(m.getElements(), 0);
	}

	results.put(layers.getOutputLayer(), outputError);
	this.activations = activations;

	calculate(calculatedLayers, results, layers.getInputLayer());
    }

    @Override
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target) {
	if (activation.getElements().length != target.getElements().length || activation.getColumns() != target.getColumns()) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	Matrix result = new Matrix(activation);
	for (int i = 0; i < activation.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - activation.getElements()[i]) * activation.getElements()[i] * (activation.getElements()[i] - 1);
	}

	return result;
    }

    private static class AparapiRmsDerivativeByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] activation;
	private float[] weightUpdates;
	private float learningRate;
	private float momentum;

	@Override
	protected void init(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	    super.init(graph, inputMatrix, outputMatrix);
	    if (weightUpdates == null || weightUpdates.length != graph.getConnectionGraph().getElements().length) {
		weightUpdates = new float[graph.getConnectionGraph().getElements().length];
	    }
	}

	@Override
	protected void outputCalculated(int outputIndex) {
	    int id = getGlobalId();
	    for (int i = 0; i < inputOutputColumns; i++) {
		int outputIdx = (outputStartIndex + id) * inputOutputColumns + i;
		output[outputIdx] *= activation[outputIdx] * (activation[outputIdx] - 1);

		int weightStart = id * weightsColumns;
		for (int j = 0; j < weightsColumns; j++) {
		    int weightIdx = weightStart + j;
		    float currentUpdate = learningRate * output[outputIdx] * input[(inputStartIndex + j) * inputOutputColumns + i] + momentum * weightUpdates[weightIdx];
		    weights[weightIdx] += currentUpdate;
		    weightUpdates[weightIdx] = currentUpdate;
		}
	    }
	}
    }

    private static class AparapiRmsDerivativeByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] activation;
	private float[] weightUpdates;
	private float learningRate;
	private float momentum;

	@Override
	protected void init(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	    super.init(graph, inputMatrix, outputMatrix);
	    if (weightUpdates == null || weightUpdates.length != graph.getConnectionGraph().getElements().length) {
		weightUpdates = new float[graph.getConnectionGraph().getElements().length];
	    }
	}

	@Override
	protected void outputCalculated(int outputIndex) {
	    int id = getGlobalId();
	    for (int i = 0; i < inputOutputColumns; i++) {
		int outputIdx = (inputStartIndex + id) * inputOutputColumns + i;
		output[outputIdx] *= activation[outputIdx] * (activation[outputIdx] - 1);

		for (int j = 0; j < weightsRows; j++) {
		    int weightIdx = j * weightsColumns + id;
		    float currentUpdate = learningRate * output[outputIdx] * input[(outputStartIndex + j) * inputOutputColumns + i] + momentum * weightUpdates[weightIdx];
		    weights[weightIdx] += currentUpdate;
		    weightUpdates[weightIdx] = currentUpdate;
		}
	    }
	}
    }
}
