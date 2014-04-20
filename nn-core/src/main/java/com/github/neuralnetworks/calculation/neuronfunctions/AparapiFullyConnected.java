package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;
import java.util.stream.IntStream;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.Util;

/**
 * Base Aparapi connection calculator for fully connected layers.
 * If there are multiple inbound connections they are combined
 * in a "single" connection and are calculated simultaneously
 * 
 * !!! IMPORTANT !!! Aparapi only works one-dimensional arrays of primitive data
 * types can only call member methods of the Kernel class itself.
 * 
 * Because of this limitations all the data that is contained in the input
 * connections, weight matrices, input values etc is converted into
 * one-dimensional member arrays of this class
 */
public abstract class AparapiFullyConnected extends Kernel implements ConnectionCalculator {

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
    protected float[] input;
    @Constant
    protected final int[] inputStartPositions;
    @Constant
    protected final int[] inputRowSteps;
    @Constant
    protected final int[] inputColumnSteps;

    /**
     * output values
     */
    protected float[] output;
    protected final int outputStartPosition;
    protected final int outputRowStep;
    protected final int outputColumnStep;

    /**
     * This is combined with the other properties to represent the
     * FullyConnected connection (the FullyConnected class itself cannot be used
     * because of the Aparapi limitations) It is an array, because of the
     * combined connections
     */
    protected final float[] weights;
    @Constant
    protected final int[] weightStartPositions;
    @Constant
    protected final int[] weightsSize;
    @Constant
    protected final int[] weightsInitialStep;
    @Constant
    protected final int[] weightsStep;

    public AparapiFullyConnected(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super();
	this.miniBatchSize = TensorFactory.batchSize(valuesProvider);

	// input
	input = TensorFactory.tensor(Util.getOppositeLayer(inputConnections.get(0), targetLayer), inputConnections.get(0), valuesProvider).getElements();
	weights = ((FullyConnected) inputConnections.get(0)).getWeights().getElements();
	inputConnections.forEach(c -> {
	    Tensor t = TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider);
	    if (!(c instanceof FullyConnected)) {
		throw new IllegalArgumentException("Only FullyConnected connections are supported");
	    }

	    if (!(t instanceof Matrix)) {
		throw new IllegalArgumentException("Only matrices are supported as input");
	    }

	    if (input != t.getElements()) {
		throw new IllegalArgumentException("Only one input array is allowed");
	    }

	    if (weights != ((FullyConnected) c).getWeights().getElements()) {
		throw new IllegalArgumentException("Only one weight array is allowed");
	    }
	});

	this.series = inputConnections.size();
	this.inputStartPositions = new int[series];
	this.inputRowSteps = new int[series];
	this.inputColumnSteps = new int[series];
	IntStream.range(0, inputConnections.size()).forEach(i -> {
	    Matrix m = TensorFactory.tensor(Util.getOppositeLayer(inputConnections.get(i), targetLayer), inputConnections.get(i), valuesProvider);
	    inputStartPositions[i] = m.getStartIndex();
	    inputRowSteps[i] = m.getRowElementsDistance();
	    inputColumnSteps[i] = m.getColumnElementsDistance();
	});

	// output
	Matrix o = TensorFactory.tensor(targetLayer, inputConnections, valuesProvider);
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
	    Matrix w = ((FullyConnected) inputConnections.get(i)).getWeights();
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

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (accept(connections, valuesProvider, targetLayer)) {
	    Environment.getInstance().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
	} else {
	    throw new IllegalArgumentException("A parameter does not match");
	}
    }

    public boolean accept(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (TensorFactory.batchSize(valuesProvider) != miniBatchSize) {
	    return false;
	}

	if (TensorFactory.tensor(targetLayer, connections, valuesProvider).getElements() != output) {
	    return false;
	}

	if (connections.size() != series || connections.size() == 0) {
	    return false;
	}

	if (connections.stream().filter(c -> TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider).getElements() != input).findAny().isPresent()) {
	    return false;
	}

	return true;
    }

    public float[] getInput() {
        return input;
    }

    public void setInput(float[] input) {
        this.input = input;
    }

    public float[] getOutput() {
        return output;
    }

    public void setOutput(float[] output) {
        this.output = output;
    }
}
