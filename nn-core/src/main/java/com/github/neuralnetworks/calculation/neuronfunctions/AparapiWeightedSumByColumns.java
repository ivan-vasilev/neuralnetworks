package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;

public class AparapiWeightedSumByColumns extends AparapiBaseFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected int[] weightsRows;

    @Override
    public void run() {
	int id = getGlobalId();

	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    for (int j = 0; j < weightsRows[0]; j++) {
		output[outputIndex(id, i, 0)] += input[inputIndex(j, i, 0)] * weights[weightIndex(j, id, 0)];
	    }

//	    if (series >= 2) {
//		for (int j = 0; j < weightsRows[1]; j++) {
//		    output[outputIndex(id, i, 1)] += input1[inputIndex(j, i, 1)] * weights1[weightIndex(j, id, 1)];
//		}
//	    }
//
//	    if (series >= 3) {
//		for (int j = 0; j < weightsRows[2]; j++) {
//		    output[outputIndex(id, i, 2)] += input2[inputIndex(j, i, 2)] * weights2[weightIndex(j, id, 2)];
//		}
//	    }

	    after(id, i);
	}
    }

    /**
     * initialization before the actual calculation
     */
    @Override
    protected void init(Map<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	super.init(input, outputMatrix, targetLayer);

	weightsRows = new int[series];

	int i = 0;
	for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
	    Connections graph = e.getKey();
	    Matrix cg = graph.getConnectionGraph();
	    Matrix inputMatrix = e.getValue();

	    if (inputMatrix.getColumns() != outputMatrix.getColumns() || (outputMatrix.getRows() != cg.getColumns() && cg.getColumns() != 1)) {
		throw new IllegalArgumentException("matrices do not match");
	    }

	    weightsRows[i++] = graph instanceof OneToOne ? 1 : cg.getRows();
	}
    }

    protected void before(int row, int column) {
    }

    protected void after(int row, int column) {
    }
}
