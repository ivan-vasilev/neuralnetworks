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

public class BackPropagationAparapiImpl implements BackPropagation {

    private BackPropagationLayerCalculator layerCalculator;
    private Map<Layer, Matrix> results;

    public BackPropagationAparapiImpl(float learningRate) {
	super();
	this.layerCalculator = new BackPropagationLayerCalculator(learningRate);
	this.results = new HashMap<Layer, Matrix>();
    }

    @Override
    public void backPropagate(Matrix outputError, InputOutputLayers layers) {
	Set<Layer> calculatedLayers = new HashSet<Layer>();
	calculatedLayers.add(layers.getOutputLayer());
	for (Matrix m : results.values()) {
	    Util.fillArray(m.getElements(), 0);
	}

	results.put(layers.getOutputLayer(), outputError);
	layerCalculator.calculate(calculatedLayers, results, layers.getInputLayer());
    }

    @Override
    public Matrix getOutputErrorDerivative(Matrix actual, Matrix target) {
	if (actual.getElements().length != target.getElements().length || actual.getColumns() != target.getColumns()) {
	    throw new IllegalArgumentException("Matrices are not the same");
	}

	Matrix result = new Matrix(actual);
	for (int i = 0; i < actual.getElements().length; i++) {
	    result.getElements()[i] = (target.getElements()[i] - actual.getElements()[i]) * actual.getElements()[i] * (actual.getElements()[i] - 1);
	}

	return result;
    }

    private static class BackPropagationLayerCalculator extends LayerCalculatorImpl {

	private static final long serialVersionUID = 1L;

	private final AparapiRmsDerivativeByRows forward;
	private final AparapiRmsDerivativeByColumns backward;

	public BackPropagationLayerCalculator(float learningRate) {
	    super();
	    forward = new AparapiRmsDerivativeByRows();
	    backward = new AparapiRmsDerivativeByColumns();
	    forward.learningRate = backward.learningRate = learningRate;
	}

	@Override
	protected InputFunction getInputFunction(Connections connection, Layer layer) {
	    return connection.getInputLayer() != layer ? forward : backward;
	}
    }

    private static class AparapiRmsDerivativeByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] actual;
	private float learningRate;

	@Override
	protected void outputCalculated(int outputIndex) {
	    int id = getGlobalId();
	    for (int i = 0; i < inputOutputColumns; i++) {
		int outputIdx = (outputStartIndex + id) * inputOutputColumns + i;
		output[outputIdx] *= actual[outputIdx] * (actual[outputIdx] - 1);

		int weightStart = id * weightsColumns;
		for (int j = 0; j < weightsColumns; j++) {
		    weights[weightStart + j] += learningRate * output[outputIdx] * actual[outputIdx];
		}
	    }
	}
    }

    private static class AparapiRmsDerivativeByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] actual;
	private float learningRate;

	@Override
	protected void outputCalculated(int outputIndex) {
	    int id = getGlobalId();
	    for (int i = 0; i < inputOutputColumns; i++) {
		int outputIdx = (inputStartIndex + id) * inputOutputColumns + i;
		output[outputIdx] *= actual[outputIdx] * (actual[outputIdx] - 1);

		for (int j = 0; j < weightsRows; j++) {
		    weights[j * weightsColumns + id] += learningRate * output[outputIdx] * actual[outputIdx];
		}
	    }
	}
    }
}
