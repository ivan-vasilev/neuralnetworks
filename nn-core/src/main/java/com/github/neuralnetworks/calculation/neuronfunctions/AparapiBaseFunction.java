package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Iterator;
import java.util.Map;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * base class for input functions
 */
public abstract class AparapiBaseFunction extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = -8435155322138790083L;

    protected int inputOutputColumns;
    protected int series;
    protected float[] output;
    protected int[] weightsColumns;
    protected int[] inputStartIndexes;
    protected int[] outputStartIndexes;

    protected float[] weights;
    protected float[] input;

    protected float[] weights1;
    protected float[] input1;

    protected float[] weights2;
    protected float[] input2;

    @Override
    public void calculate(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	if (input.size() > 0) {
	    init(input, outputMatrix, targetLayer);
	    execute(outputMatrix.getRows());
	}
    }

    /**
     * initialization before the actual calculation
     */
    protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	boolean hasInput = false, hasOutput = false;

	Iterator<Matrix> it = input.values().iterator();
	this.inputOutputColumns = it.next().getColumns();
	while (it.hasNext()) {
	    if (inputOutputColumns != it.next().getColumns()) {
		throw new IllegalArgumentException("Input is not the same");
	    }
	}

	this.weightsColumns = new int[input.size()];
	this.inputStartIndexes = new int[input.size()];
	this.outputStartIndexes = new int[input.size()];
	this.output = outputMatrix.getElements();

	this.series = 0;
	for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
	    Connections graph = e.getKey();
	    Matrix inputMatrix = e.getValue();
	    Matrix cg = graph.getConnectionGraph();

	    this.weightsColumns[series] = cg.getColumns();
	    this.inputStartIndexes[series] = graph.getInputLayerStartNeuron();
	    this.outputStartIndexes[series] = graph.getOutputLayerStartNeuron();

	    switch (series) {
	    case 0:
		this.input = inputMatrix.getElements();
		this.weights = cg.getElements();
		break;

	    case 1:
		this.input1 = inputMatrix.getElements();
		this.weights1 = cg.getElements();
		break;

	    case 2:
		this.input2 = inputMatrix.getElements();
		this.weights2 = cg.getElements();
		break;
	    }

	    series++;

	    if (!(graph instanceof OneToOne)) {
		if (graph.getInputLayer() == targetLayer) {
		    hasInput = true;
		}
		
		if (graph.getOutputLayer() == targetLayer) {
		    hasOutput = true;
		}
		
		if (hasInput && hasOutput) {
		    throw new IllegalArgumentException("Functions must only be for input or output layer, but not both");
		}
	    }
	}

	if (series < 2) {
	    this.weights1 = new float[1]; // 1 for aparapi reasons
	    this.input1 = new float[1];
	}

	if (series < 3) {
	    this.weights2 = new float[1];
	    this.input2 = new float[1];
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    };

    protected int weightIndex(int row, int column, int series) {
	return row * weightsColumns[series] + column;
    }

    protected int inputIndex(int row, int column, int series) {
	return (inputStartIndexes[series] + row) * inputOutputColumns + column;
    }

    protected int outputIndex(int row, int column, int series) {
	return (outputStartIndexes[series] + row) * inputOutputColumns + column;
    }

    protected int outputBaseIndex(int row, int column) {
	return row * inputOutputColumns + column;
    }
}
