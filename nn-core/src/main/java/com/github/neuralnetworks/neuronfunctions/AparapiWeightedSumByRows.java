package com.github.neuralnetworks.neuronfunctions;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.AparapiExecutionMode;

public class AparapiWeightedSumByRows extends Kernel implements InputFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected float weights[];
    protected float[] input;
    protected float[] output;
    protected int weightsColumns;
    protected int inputOutputColumns;
    protected int inputStartIndex;
    protected int outputStartIndex;

    @Override
    public void run() {
	int id = getGlobalId();
	for (int i = 0; i < inputOutputColumns; i++) {
	    int outputIdx = (outputStartIndex + id) * inputOutputColumns + i;
	    int weightsStartIdx = weightsColumns * id;
	    for (int j = 0; j < weightsColumns; j++) {
		output[outputIdx] += input[(inputStartIndex + j) * inputOutputColumns + i] * weights[weightsStartIdx + j];
	    }

	    outputCalculated(outputIdx);
	}
    }

    @Override
    public void calculate(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	Matrix cg = graph.getConnectionGraph();

	if (inputMatrix.getColumns() != outputMatrix.getColumns() || outputMatrix.getRows() != cg.getRows()) {
	    throw new IllegalArgumentException("matrices do not match");
	}

	this.weights = cg.getElements();
	this.input = inputMatrix.getElements();
	this.output = outputMatrix.getElements();

	this.weightsColumns = cg.getColumns();
	this.inputOutputColumns = inputMatrix.getColumns();
	this.outputStartIndex = graph.getOutputLayerStartNeuron();
	this.inputStartIndex = graph.getInputLayerStartNeuron();

	setExecutionMode(AparapiExecutionMode.getInstance().getExecutionMode());
	this.execute(outputMatrix.getRows());
    }

    /**
     * @param outputIndex - index within the ouptut array
     */
    protected void outputCalculated(int outputIndex) {
    }
}
