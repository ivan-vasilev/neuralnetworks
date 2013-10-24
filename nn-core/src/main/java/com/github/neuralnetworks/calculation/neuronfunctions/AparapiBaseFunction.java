package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * base class for input functions
 */
public abstract class AparapiBaseFunction extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = -8435155322138790083L;

    protected int inputOutputColumns;
    protected float[] output;
    protected int series;

    protected float[] weights;
    protected float[] input;
    protected int weightsColumns;
    protected int inputStartIndex;
    protected int outputStartIndex;

    protected float[] weights1;
    protected float[] input1;
    protected int weightsColumns1;
    protected int inputStartIndex1;
    protected int outputStartIndex1;

    protected float[] weights2;
    protected float[] input2;
    protected int weightsColumns2;
    protected int inputStartIndex2;
    protected int outputStartIndex2;

    @Override
    public void calculate(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	init(input, outputMatrix, targetLayer);
	execute(outputMatrix.getRows());
    }

    /**
     * initialization before the actual calculation
     */
    protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	boolean hasInput = false, hasOutput = false;

	for (Connections c : input.keySet()) {
	    if (c.getInputLayer() == targetLayer) {
		hasInput = true;
	    }

	    if (c.getOutputLayer() == targetLayer) {
		hasOutput = true;
	    }

	    if (hasInput && hasOutput) {
		throw new IllegalArgumentException("Functions must only be for input or output layer, but not both");
	    }
	}

	Iterator<Matrix> it = input.values().iterator();
	this.inputOutputColumns = it.next().getColumns();
	while (it.hasNext()) {
	    if (inputOutputColumns != it.next().getColumns()) {
		throw new IllegalArgumentException("Input is not the same");
	    }
	}

	this.series = 0;
	this.output = outputMatrix.getElements();
	this.inputOutputColumns = 0;

	for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
	    Connections graph = e.getKey();
	    Matrix inputMatrix = e.getValue();
	    Matrix cg = graph.getConnectionGraph();

	    switch (series) {
	    case 0:
		this.weights = cg.getElements();
		this.input = inputMatrix.getElements();

		this.weightsColumns = cg.getColumns();
		this.inputOutputColumns = inputMatrix.getColumns();
		this.inputStartIndex = graph.getInputLayerStartNeuron();
		this.outputStartIndex = graph.getOutputLayerStartNeuron();
		break;

	    case 1:
		this.weights1 = cg.getElements();
		this.input1 = inputMatrix.getElements();

		this.weightsColumns1 = cg.getColumns();
		this.inputStartIndex1 = graph.getInputLayerStartNeuron();
		this.outputStartIndex1 = graph.getOutputLayerStartNeuron();
		break;

	    case 2:
		this.weights2 = cg.getElements();
		this.input2 = inputMatrix.getElements();

		this.weightsColumns2 = cg.getColumns();
		this.inputStartIndex2 = graph.getInputLayerStartNeuron();
		this.outputStartIndex2 = graph.getOutputLayerStartNeuron();
		break;
	    }

	    series++;
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    };

    protected int weightIndex(int row, int column, int series) {
	int weightsColumns = 0;
	if (series == 0) {
	    weightsColumns = this.weightsColumns;
	} else if (series == 1) {
	    weightsColumns = this.weightsColumns1;
	} else if (series == 1) {
	    weightsColumns = this.weightsColumns2;
	}

	return row * weightsColumns + column;
    }

    protected int inputIndex(int row, int column, int series) {
	int inputStartIndex = 0;
	if (series == 0) {
	    inputStartIndex = this.inputStartIndex;
	} else if (series == 1) {
	    inputStartIndex = this.inputStartIndex1;
	} else if (series == 1) {
	    inputStartIndex = this.inputStartIndex2;
	}

	return (inputStartIndex + row) * inputOutputColumns + column;
    }

    protected int outputIndex(int row, int column, int series) {
	int outputStartIndex = 0;
	if (series == 0) {
	    outputStartIndex = this.outputStartIndex;
	} else if (series == 1) {
	    outputStartIndex = this.outputStartIndex1;
	} else if (series == 1) {
	    outputStartIndex = this.outputStartIndex2;
	}

	return (outputStartIndex + row) * inputOutputColumns + column;
    }

    protected int outputBaseIndex(int row, int column) {
	return row * inputOutputColumns + column;
    }

    public Layer getTargetLayer(Collection<Connections> c) {
	Layer result = null;
	return result;
    }
}
