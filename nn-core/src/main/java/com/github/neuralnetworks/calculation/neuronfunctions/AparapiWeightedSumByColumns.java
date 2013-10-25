package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;

public class AparapiWeightedSumByColumns extends AparapiBaseFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected int weightsRows;
    protected int weightsRows1;
    protected int weightsRows2;

    @Override
    public void run() {
	int id = getGlobalId();

	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    for (int j = 0; j < weightsRows; j++) {
		output[outputIndex(id, i, 0)] += input[inputIndex(j, i, 0)] * weights[weightIndex(j, id, 0)];
	    }

	    if (series >= 1) {
		for (int j = 0; j < weightsRows1; j++) {
		    output[outputIndex(id, i, 1)] += input1[inputIndex(j, i, 1)] * weights1[weightIndex(j, id, 1)];
		}
	    }

	    if (series >= 2) {
		for (int j = 0; j < weightsRows2; j++) {
		    output[outputIndex(id, i, 2)] += input2[inputIndex(j, i, 2)] * weights2[weightIndex(j, id, 2)];
		}
	    }

	    after(id, i);
	}
    }

    /**
     * initialization before the actual calculation
     */
    @Override
    protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	super.init(input, outputMatrix, targetLayer);

	int i = 0;
	for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
	    Connections graph = e.getKey();
	    Matrix cg = graph.getConnectionGraph();
	    Matrix inputMatrix = e.getValue();

	    if (inputMatrix.getColumns() != outputMatrix.getColumns() || (outputMatrix.getRows() != cg.getColumns() && cg.getColumns() != 1)) {
		throw new IllegalArgumentException("matrices do not match");
	    }

	    switch (i) {
	    case 0:
		this.weightsRows = graph instanceof OneToOne ? 1 : cg.getRows();
		break;
	    case 1:
		this.weightsRows1 = graph instanceof OneToOne ? 1 : cg.getRows();
		break;
	    case 2:
		this.weightsRows2 = graph instanceof OneToOne ? 1 : cg.getRows();
		break;
	    }

	    i++;
	}

	if (series < 2) {
	    weightsRows1 = 0;
	}

	if (series < 3) {
	    weightsRows2 = 0;
	}
    }

    protected void before(int row, int column) {
    }

    protected void after(int row, int column) {
    }
}
