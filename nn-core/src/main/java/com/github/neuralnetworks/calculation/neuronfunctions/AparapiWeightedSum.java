package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;
import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Util;

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
    protected final int miniBatchSize;

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
    protected final float[] weights;

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
    @Constant
    protected final int[] weightsDimension;

    /**
     * For optimization reasons
     */
    //@Local TODO
    @Constant
    protected final int[] weightsInitialStep;

    /**
     * For optimization reasons
     */
    //@Local TODO
    @Constant
    protected final int[] weightsStep;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    //@Local TODO
    @Constant
    protected final int[] inputStartPositions;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    //@Local TODO
    @Constant
    protected final int[] weightStartPositions;

    /**
     * Will determine whether initialization is needed
     */
    protected Layer currentLayer;

    public AparapiWeightedSum(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer) {
	super();

	this.currentLayer = targetLayer;
	this.miniBatchSize = miniBatchSize;
	this.series = inputConnections.size();
	this.weightsDimension = new int[series];
	this.inputStartPositions = new int[series];
	this.weightStartPositions = new int[series];
	this.weightsInitialStep = new int[series];
	this.weightsStep = new int[series];

	int totalInputSize = 0, totalWeightSize = 0, i = 0;
	for (java.util.Map.Entry<GraphConnections, Integer> e : inputConnections.entrySet()) {
	    Matrix cg = e.getKey().getConnectionGraph();

	    inputStartPositions[i] = totalInputSize;
	    totalInputSize += e.getValue();
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
	    java.util.Map.Entry<GraphConnections, Integer> e = inputConnections.entrySet().iterator().next();
	    this.weights = e.getKey().getConnectionGraph().getElements();
	} else {
	    weights = new float[totalWeightSize];
	    this.input = new float[totalInputSize];

	    i = 0;
	    for (java.util.Map.Entry<GraphConnections, Integer> e : inputConnections.entrySet()) {
		System.arraycopy(e.getKey().getConnectionGraph().getElements(), 0, weights, weightStartPositions[i], e.getKey().getConnectionGraph().getElements().length);
		i++;
	    }
	}
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (connections.size() > 0) {
	    init(connections, valuesProvider, targetLayer);
	    
	    Environment.getInstance().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
	}
    }

    /**
     * Combines all the inputConnections and initializes all the arrays based on
     * the connections
     */
    protected void init(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	this.output = valuesProvider.getValues(targetLayer, inputConnections).getElements();

	if (inputConnections.size() == 1) {
	    this.input = valuesProvider.getValues(Util.getOppositeLayer(inputConnections.get(0), targetLayer), inputConnections).getElements();
	} else {
	    int offset = 0;
	    for (Connections c: inputConnections) {
		float[] a = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c).getElements();
		System.arraycopy(a, 0, input, offset, a.length);
		offset += a.length;
	    }
	}
    };

    @Override
    public void run() {
	int id = getGlobalId();

	int inputStartPosition = 0, initialWeightIndex = 0, weightStep = 0, dim = 0;
	float value = 0;

	// each input example
	for (int i = 0; i < miniBatchSize; i++) {
	    // each connection (of the combined connections)
	    value = output[id * miniBatchSize + i];
	    for (int k = 0; k < series; k++) {
		// each element in the row/column
		inputStartPosition = inputStartPositions[k];
		initialWeightIndex = weightStartPositions[k] + weightsInitialStep[k] * id;
		weightStep = weightsStep[k];
		dim = weightsDimension[k];

		for (int j = 0; j < dim; j++) {
		    value += input[inputStartPosition + j * miniBatchSize + i] * weights[initialWeightIndex + j * weightStep];
		}
	    }

	    output[id * miniBatchSize + i] = value;
	}

	after();
    }

    protected void after() {
    }
}
