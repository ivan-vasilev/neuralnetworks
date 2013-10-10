package com.github.neuralnetworks.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;
import com.github.neuralnetworks.util.AparapiExecutionMode;

public class AparapiWeightedSumByColumns extends Kernel implements InputFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected float weights[];
    protected float[] input;
    protected float[] output;
    protected int weightsColumns;
    protected int weightsRows;
    protected int inputOutputColumns;
    protected int inputStartIndex;
    protected int outputStartIndex;

    @Override
    public void run() {
	int id = getGlobalId();
	for (int i = 0; i < inputOutputColumns; i++) {
	    int outputIdx = (inputStartIndex + id) * inputOutputColumns + i;
	    for (int j = 0; j < weightsRows; j++) {
		output[outputIdx] += input[(outputStartIndex + j) * inputOutputColumns + i] * weights[weightsColumns * j + id];
	    }

	    outputCalculated(outputIdx);
	}
    }

    @Override
    public void calculate(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	Matrix cg = graph.getConnectionGraph();

	if (inputMatrix.getColumns() != outputMatrix.getColumns() || (outputMatrix.getRows() != cg.getColumns() && cg.getColumns() != 1)) {
	    throw new IllegalArgumentException("matrices do not match");
	}

	this.weights = cg.getElements();
	this.input = inputMatrix.getElements();
	this.output = outputMatrix.getElements();

	this.weightsColumns = cg.getColumns();
	this.weightsRows = graph instanceof OneToOne ? 1 : cg.getRows();
	this.inputOutputColumns = inputMatrix.getColumns();
	this.outputStartIndex = graph.getOutputLayerStartNeuron();
	this.inputStartIndex = graph.getInputLayerStartNeuron();

	setExecutionMode(AparapiExecutionMode.getInstance().getExecutionMode());
	this.execute(outputMatrix.getRows());
    }

    /**
     * @param outputIndex
     *            - index within the output array
     */
    protected void outputCalculated(int outputIndex) {
    }
}
