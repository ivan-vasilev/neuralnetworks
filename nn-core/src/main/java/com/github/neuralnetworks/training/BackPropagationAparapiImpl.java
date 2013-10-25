package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.InputOutputLayers;
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
    public void backPropagate(Map<Layer, Matrix> activations, Matrix outputError, InputOutputLayers layers) {
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
    public void calculate(Map<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Map<Connections, Matrix> forward = new HashMap<>();
	Map<Connections, Matrix> backward = new HashMap<>();
	Map<Connections, Matrix> bias = new HashMap<>();
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
	private float[] weightUpdates1;
	private float[] weightUpdates2;
	private Map<Connections, float[]> storedWeightUpdates = new HashMap<>();
	private float learningRate;
	private float momentum;
	private Map<Layer, Matrix> activations;

	@Override
	protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);

	    int i = 0;
	    for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
		Connections graph = e.getKey();

		switch (i) {
		case 0:
		    weightUpdates = storedWeightUpdates.get(graph);
		    if (weightUpdates == null) {
			weightUpdates = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates);
		    }
		    break;
		case 1:
		    weightUpdates1 = storedWeightUpdates.get(graph);
		    if (weightUpdates1 == null) {
			weightUpdates1 = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates1);
		    }
		    break;
		case 2:
		    weightUpdates2 = storedWeightUpdates.get(graph);
		    if (weightUpdates2 == null) {
			weightUpdates2 = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates2);
		    }
		    break;
		}

		i++;
	    }

	    if (series <= 2) {
		weightUpdates1 = new float[1];
	    }
	    
	    if (series <= 3) {
		weightUpdates2 = new float[1];
	    }

	    outputActivation = activations.get(targetLayer).getElements();
	}

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);

	    for (int j = 0; j < weightsColumns; j++) {
		float weightUpdate = learningRate * input[inputIndex(j, column, 0)] * outputActivation[outputIndex(row, column, 0)] + momentum * weightUpdates[weightIndex(row, j, 0)];
		weights[weightIndex(row, j, 0)] += weightUpdate;
		weightUpdates[weightIndex(row, j, 0)] = weightUpdate;
	    }

	    if (series >= 1) {
		for (int j = 0; j < weightsColumns1; j++) {
		    float weightUpdate = learningRate * input1[inputIndex(j, column, 1)] * outputActivation[outputIndex(row, column, 1)] + momentum * weightUpdates1[weightIndex(row, j, 1)];
		    weights1[weightIndex(row, j, 1)] += weightUpdate;
		    weightUpdates1[weightIndex(row, j, 1)] = weightUpdate;
		}
	    }

	    if (series >= 2) {
		for (int j = 0; j < weightsColumns2; j++) {
		    float weightUpdate = learningRate * input2[inputIndex(j, column, 2)] * outputActivation[outputIndex(row, column, 2)] + momentum * weightUpdates2[weightIndex(row, j, 2)];
		    weights2[weightIndex(row, j, 2)] += weightUpdate;
		    weightUpdates2[weightIndex(row, j, 2)] = weightUpdate;
		}
	    }
	}
    }

    private static class AparapiRmsDerivativeByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -5101971690861270462L;

	private float[] outputActivation;
	private Map<Connections, float[]> storedWeightUpdates = new HashMap<>();
	private float[] weightUpdates;
	private float[] weightUpdates1;
	private float[] weightUpdates2;
	private float learningRate;
	private float momentum;
	private Map<Layer, Matrix> activations;

	@Override
	protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);

	    int i = 0;
	    for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
		Connections graph = e.getKey();

		switch (i) {
		case 0:
		    weightUpdates = storedWeightUpdates.get(graph);
		    if (weightUpdates == null) {
			weightUpdates = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates);
		    }
		    break;
		case 1:
		    weightUpdates1 = storedWeightUpdates.get(graph);
		    if (weightUpdates1 == null) {
			weightUpdates1 = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates1);
		    }
		    break;
		case 2:
		    weightUpdates2 = storedWeightUpdates.get(graph);
		    if (weightUpdates2 == null) {
			weightUpdates2 = new float[graph.getConnectionGraph().getElements().length];
			storedWeightUpdates.put(graph, weightUpdates2);
		    }
		}

		i++;
	    }

	    if (series <= 2) {
		weightUpdates1 = new float[1];
	    }
	    
	    if (series <= 3) {
		weightUpdates2 = new float[1];
	    }

	    outputActivation = activations.get(targetLayer).getElements();
	}

	@Override
	protected void after(int row, int column) {
	    output[outputBaseIndex(row, column)] *= outputActivation[outputBaseIndex(row, column)] * (1 - outputActivation[outputBaseIndex(row, column)]);

	    for (int j = 0; j < weightsRows; j++) {
		float weightUpdate = learningRate * input[inputIndex(j, column, 0)] * outputActivation[outputIndex(row, column, 0)] + momentum * weightUpdates[weightIndex(j, row, 0)];
		weights[weightIndex(j, row, 0)] += weightUpdate;
		weightUpdates[weightIndex(j, row, 0)] = weightUpdate;
	    }

	    if (series >= 1) {
		for (int j = 0; j < weightsRows1; j++) {
		    float weightUpdate = learningRate * input1[inputIndex(j, column, 1)] * outputActivation[outputIndex(row, column, 1)] + momentum * weightUpdates1[weightIndex(j, row, 1)];
		    weights1[weightIndex(j, row, 1)] += weightUpdate;
		    weightUpdates1[weightIndex(j, row, 1)] = weightUpdate;
		}
	    }

	    if (series >= 2) {
		for (int j = 0; j < weightsRows2; j++) {
		    float weightUpdate = learningRate * input2[inputIndex(j, column, 2)] * outputActivation[outputIndex(row, column, 2)] + momentum * weightUpdates2[weightIndex(j, row, 2)];
		    weights2[weightIndex(j, row, 2)] += weightUpdate;
		    weightUpdates2[weightIndex(j, row, 2)] = weightUpdate;
		}
	    }
	}
    }
}