package com.github.neuralnetworks.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.AparapiExecutionMode;

/**
 * base class for input functions
 */
public abstract class AparapiBaseFunction extends Kernel implements InputFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected float[] weights;
    protected float[] input;
    protected float[] output;
    protected int weightsColumns;
    protected int inputOutputColumns;
    protected int inputStartIndex;
    protected int outputStartIndex;

    @Override
    public void calculate(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	init(graph, inputMatrix, outputMatrix);
	execute(outputMatrix.getRows());
    }

    /**
     * initialization before the actual calculation
     */
    protected void init(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	Matrix cg = graph.getConnectionGraph();

	this.weights = cg.getElements();
	this.input = inputMatrix.getElements();
	this.output = outputMatrix.getElements();

	this.weightsColumns = cg.getColumns();
	this.inputOutputColumns = inputMatrix.getColumns();
	this.inputStartIndex = graph.getInputLayerStartNeuron();
	this.outputStartIndex = graph.getOutputLayerStartNeuron();

	setExecutionMode(AparapiExecutionMode.getInstance().getExecutionMode());
    };

    protected int weightIndex(int row, int column) {
	return row * weightsColumns + column;
    }

    protected int inputIndex(int row, int column) {
	return (inputStartIndex + row) * inputOutputColumns + column;
    }

    protected int outputIndex(int row, int column) {
	return (outputStartIndex + row) * inputOutputColumns + column;
    }
}
