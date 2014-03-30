package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;
import java.util.stream.IntStream;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
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
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    @Constant
    protected final float[] input;
    @Constant
    protected final int[] inputStartPositions;
    @Constant
    protected final int[] inputRowSteps;
    @Constant
    protected final int[] inputColumnSteps;

    /**
     * output values
     */
    protected final float[] output;
    protected final int outputStartPosition;
    protected final int outputRowStep;
    protected final int outputColumnStep;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    @Constant
    protected final float[] weights;
    @Constant
    protected final int[] weightStartPositions;
    @Constant
    protected final int[] weightsSize;
    @Constant
    protected final int[] weightsInitialStep;
    @Constant
    protected final int[] weightsStep;

    public AparapiWeightedSum(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super();
	this.miniBatchSize = valuesProvider.getMiniBatchSize();

	// input
	input = valuesProvider.getValues(Util.getOppositeLayer(inputConnections.get(0), targetLayer), inputConnections.get(0)).getElements();
	weights = ((FullyConnected) inputConnections.get(0)).getConnectionGraph().getElements();
	inputConnections.forEach(c -> {
	    Tensor t = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c);
	    if (!(c instanceof FullyConnected)) {
		throw new IllegalArgumentException("Only FullyConnected connections are supported");
	    }

	    if (!(t instanceof Matrix)) {
		throw new IllegalArgumentException("Only matrices are supported as input");
	    }

	    if (input != t.getElements()) {
		throw new IllegalArgumentException("Only one input array is allowed");
	    }

	    if (weights != ((FullyConnected) c).getConnectionGraph().getElements()) {
		throw new IllegalArgumentException("Only one input weight array is allowed");
	    }
	});

	this.series = inputConnections.size();
	this.inputStartPositions = new int[series];
	this.inputRowSteps = new int[series];
	this.inputColumnSteps = new int[series];
	IntStream.range(0, inputConnections.size()).forEach(i -> {
	    Matrix m = valuesProvider.getValues(Util.getOppositeLayer(inputConnections.get(i), targetLayer), inputConnections.get(i));
	    inputStartPositions[i] = m.getStartIndex();
	    inputRowSteps[i] = m.getRowElementsDistance();
	    inputColumnSteps[i] = m.getColumnElementsDistance();
	});

	// output
	Matrix o = valuesProvider.getValues(targetLayer, inputConnections);
	this.output = o.getElements();
	this.outputStartPosition = o.getStartIndex();
	this.outputRowStep = o.getRowElementsDistance();
	this.outputColumnStep = o.getColumnElementsDistance();

	// weights
	this.weightStartPositions = new int[series];
	this.weightsSize = new int[series];
	this.weightsInitialStep = new int[series];
	this.weightsStep = new int[series];

	IntStream.range(0, inputConnections.size()).forEach(i -> {
	    Matrix w = ((FullyConnected) inputConnections.get(0)).getConnectionGraph();
	    weightStartPositions[i] = w.getStartIndex();
	    if (inputConnections.get(0).getOutputLayer() == targetLayer) {
		weightsSize[i] = w.getColumns();
		weightsInitialStep[i] = w.getRowElementsDistance();
		weightsStep[i] = w.getColumnElementsDistance();
	    } else {
		weightsSize[i] = w.getRows();
		weightsInitialStep[i] = w.getColumnElementsDistance();
		weightsStep[i] = w.getRowElementsDistance();
	    }
	});
    }

//    public AparapiWeightedSum(int a, List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
//	super();
//
//	this.miniBatchSize = valuesProvider.getMiniBatchSize();
//	this.series = inputConnections.size();
//	this.weightsSize = new int[series];
//	this.inputStartPositions = new int[series];
//	this.weightStartPositions = new int[series];
//	this.weightsInitialStep = new int[series];
//	this.weightSteps = new int[series];
//
//	int totalInputSize = 0, totalWeightSize = 0, i = 0;
//	for (java.util.Map.Entry<GraphConnections, Integer> e : inputConnections.entrySet()) {
//	    Matrix cg = e.getKey().getConnectionGraph();
//
//	    inputStartPositions[i] = totalInputSize;
//	    totalInputSize += e.getValue();
//	    weightStartPositions[i] = totalWeightSize;
//	    totalWeightSize += e.getKey().getConnectionGraph().getElements().length;
//
//	    // depending on the direction of the calculation
//	    if (e.getKey().getOutputLayer() == targetLayer) {
//		weightsSize[i] = cg.getColumns();
//		weightsInitialStep[i] = cg.getColumns();
//		weightSteps[i] = 1;
//	    } else {
//		weightsSize[i] = cg.getRows();
//		weightsInitialStep[i] = 1;
//		weightSteps[i] = cg.getColumns();
//	    }
//
//	    i++;
//	}
//
//	if (inputConnections.size() == 1) {
//	    java.util.Map.Entry<GraphConnections, Integer> e = inputConnections.entrySet().iterator().next();
//	    this.weights = e.getKey().getConnectionGraph().getElements();
//	} else {
//	    weights = new float[totalWeightSize];
//	    this.input = new float[totalInputSize];
//
//	    i = 0;
//	    for (java.util.Map.Entry<GraphConnections, Integer> e : inputConnections.entrySet()) {
//		System.arraycopy(e.getKey().getConnectionGraph().getElements(), 0, weights, weightStartPositions[i], e.getKey().getConnectionGraph().getElements().length);
//		i++;
//	    }
//	}
//    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (connections.size() > 0) {
	    Environment.getInstance().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
	}
    }

    /**
     * Combines all the inputConnections and initializes all the arrays based on
     * the connections
     */
//    protected void init(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
//	//this.output = valuesProvider.getValues(targetLayer, inputConnections).getElements();
//
//	if (inputConnections.size() == 1) {
//	    this.input = valuesProvider.getValues(Util.getOppositeLayer(inputConnections.get(0), targetLayer), inputConnections).getElements();
//	} else {
//	    int offset = 0;
//	    for (Connections c: inputConnections) {
//		float[] a = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c).getElements();
//		System.arraycopy(a, 0, input, offset, a.length);
//		offset += a.length;
//	    }
//	}
//    };

    @Override
    public void run() {
	int id = getGlobalId();

	int inputStartPosition = 0, inputRowsStep = 0, inputColumnsStep = 0, weightStartPosition = 0, weightStep = 0, dim = 0;
	float value = 0;

	// each input example
	for (int i = 0; i < miniBatchSize; i++) {
	    // each connection (of the combined connections)
	    value = output[id * outputRowStep + i * outputColumnStep];
	    for (int k = 0; k < series; k++) {
		// each element in the row/column
		inputStartPosition = inputStartPositions[k];
		inputRowsStep = inputRowSteps[k];
		inputColumnsStep = inputColumnSteps[k];
		weightStartPosition = weightStartPositions[k] + weightsInitialStep[k] * id;
		weightStep = weightsStep[k];
		dim = weightsSize[k];

		for (int j = 0; j < dim; j++) {
		    value += input[inputStartPosition + j * inputRowsStep + i * inputColumnsStep] * weights[weightStartPosition + j * weightStep];
		}
	    }

	    output[id * outputRowStep + i * outputColumnStep] = value;
	}

	after();
    }

    protected void after() {
    }
}
