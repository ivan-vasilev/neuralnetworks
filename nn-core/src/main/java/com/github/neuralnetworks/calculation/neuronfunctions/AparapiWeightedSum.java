package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * Base Aparapi connection calculator for weighted sum functions (matrix
 * multiplication). If there are multiple inbound connections they are combined
 * in a "single" connection and are calculated simultaneously
 * 
 * !!! IMPORTANT !!! Aparapi only works one-dimensional arrays of primitive data
 * types can only call member methods of the Kernel class itself.
 * 
 * Because of this limitations all the data that is contained in the input
 * connections, weight matrices, input values etc is converted into
 * one-dimensional member arrays of this class
 */
public class AparapiWeightedSum extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = -8435155322138790083L;

    /**
     * Number of input samples that will be calculated simultaneously
     */
    protected final int inputOutputSamples;

    /**
     * Number of input connections that will be "combined" for simultaneous
     * calculation
     */
    protected final int series;

    /**
     * input values
     */
    protected float[] input;

    /**
     * this is the weight matrix
     */
    protected float[] weights;

    /**
     * output values
     */
    protected float[] output;

    /**
     * This is combined with the "weights" to represent the weight matrix (the
     * Matrix class itself cannot be used because of the Aparapi limitations).
     * It is an array, because of the combined connections
     */
    //@Local TODO
    protected final int[] weightsDimension;

    /**
     * For optimization reasons
     */
    //@Local TODO
    protected final int[] weightsInitialStep;

    /**
     * For optimization reasons
     */
    //@Local TODO
    protected final int[] weightsStep;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    //@Local TODO
    protected final int[] inputStartPositions;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    //@Local TODO
    protected final int[] weightStartPositions;

    /**
     * helper map to reuse existing arrays for inputs
     */
    protected Map<Integer, float[]> storedInputs = new HashMap<>();

    /**
     * helper map to reuse existing arrays for outputs
     */
    protected Map<Integer, float[]> storedWeights = new HashMap<>();

    /**
     * Will determine whether initialization is needed
     */
    protected Layer currentLayer;

    public AparapiWeightedSum(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
	super();

	this.currentLayer = targetLayer;
	this.inputOutputSamples = inputOutputSamples;
	this.series = inputConnections.size();
	this.weightsDimension = new int[series];
	this.inputStartPositions = new int[series];
	this.weightStartPositions = new int[series];
	this.weightsInitialStep = new int[series];
	this.weightsStep = new int[series];

	int totalInputSize = 0, totalWeightSize = 0, i = 0;
	for (java.util.Map.Entry<GraphConnections, Matrix> e : inputConnections.entrySet()) {
	    Matrix cg = e.getKey().getConnectionGraph();

	    inputStartPositions[i] = totalInputSize;
	    totalInputSize += e.getValue().getElements().length;
	    weightStartPositions[i] = totalWeightSize;
	    totalWeightSize += e.getKey().getConnectionGraph().getElements().length;

	    // depending on the direction of the calculation
	    if (e.getKey().getOutputLayer() == targetLayer) {
		weightsDimension[i] = cg.getColumns();
		weightsInitialStep[i] = cg.getColumns();
		weightsStep[i] = 1;
	    } else {
		weightsDimension[i] = cg.getRows();
		weightsInitialStep[i] = 1;
		weightsStep[i] = cg.getColumns();
	    }

	    i++;
	}

	if (inputConnections.size() == 1) {
	    java.util.Map.Entry<GraphConnections, Matrix> e = inputConnections.entrySet().iterator().next();
	    this.weights = e.getKey().getConnectionGraph().getElements();
	} else {
	    this.weights = storedWeights.get(totalWeightSize);
	    if (weights == null) {
		this.weights = new float[totalWeightSize];
		storedWeights.put(totalWeightSize, this.weights);
	    }

	    i = 0;
	    for (java.util.Map.Entry<GraphConnections, Matrix> e : inputConnections.entrySet()) {
		System.arraycopy(e.getKey().getConnectionGraph().getElements(), 0, weights, weightStartPositions[i], e.getKey().getConnectionGraph().getElements().length);
		i++;
	    }
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());

    }

    @SuppressWarnings("unchecked")
    @Override
    public void calculate(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	if (input.size() > 0) {
	    init((SortedMap<GraphConnections, Matrix>) ((SortedMap<?, ?>) input), outputMatrix, targetLayer);
	    execute(outputMatrix.getRows());
	}
    }

    /**
     * Combines all the inputConnections and initializes all the arrays based on
     * the connections
     */
    protected void init(SortedMap<GraphConnections, Matrix> inputConnections, Matrix outputMatrix, Layer targetLayer) {
	this.output = outputMatrix.getElements();

	if (inputConnections.size() == 1) {
	    java.util.Map.Entry<GraphConnections, Matrix> e = inputConnections.entrySet().iterator().next();
	    this.input = e.getValue().getElements();
	} else {
	    int totalInputSize = 0;
	    for (java.util.Map.Entry<GraphConnections, Matrix> e : inputConnections.entrySet()) {
		totalInputSize += e.getValue().getElements().length;
	    }

	    this.input = storedInputs.get(totalInputSize);
	    if (this.input == null) {
		this.input = new float[totalInputSize];
		storedInputs.put(totalInputSize, this.input);
	    }

	    int offset = 0;
	    for (java.util.Map.Entry<GraphConnections, Matrix> e : inputConnections.entrySet()) {
		System.arraycopy(e.getValue().getElements(), 0, input, offset, e.getValue().getElements().length);
		offset += e.getValue().getElements().length;
	    }
	}
    };

    @Override
    public void run() {
	int id = getGlobalId();

	int ios = inputOutputSamples;
	int s = series;
	float value = 0;

	// each input example
	for (int i = 0; i < ios; i++) {
	    // each connection (of the combined connections)
	    value = output[outputIndex(id, i)];
	    for (int k = 0; k < s; k++) {
		// each element in the row/column
		int inputStartPosition = inputStartPositions[k];
		int initialWeightIndex = weightStartPositions[k] + weightsInitialStep[k] * id;
		int weightStep = weightsStep[k];
		int dim = weightsDimension[k];

		for (int j = 0; j < dim; j++) {
		    value += input[inputStartPosition + j * ios + i] * weights[initialWeightIndex + j * weightStep];
		}
	    }

	    after(value, id, i);
	}
    }

    protected void after(float value, int row, int column) {
	output[outputIndex(row, column)] = value;
    }

    /**
     * helper method for retrieving output value based on row, column and series
     */
    protected int outputIndex(int row, int column) {
	return row * inputOutputSamples + column;
    }
}
