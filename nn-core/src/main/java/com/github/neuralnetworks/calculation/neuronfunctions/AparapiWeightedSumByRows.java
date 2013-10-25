package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

public class AparapiWeightedSumByRows extends AparapiBaseFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    @Override
    public void run() {
	int id = getGlobalId();

	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    for (int j = 0; j < weightsColumns[0]; j++) {
		output[outputIndex(id, i, 0)] += input[inputIndex(j, i, 0)] * weights[weightIndex(id, j, 0)];
	    }

//	    if (series >= 2) {
//		for (int j = 0; j < weightsColumns[1]; j++) {
//		    output[outputIndex(id, i, 1)] += input1[inputIndex(j, i, 1)] * weights1[weightIndex(id, j, 1)];
//		}
//	    }
//
//	    if (series >= 3) {
//		for (int j = 0; j < weightsColumns[2]; j++) {
//		    output[outputIndex(id, i, 2)] += input2[inputIndex(j, i, 2)] * weights2[weightIndex(id, j, 2)];
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

	for (java.util.Map.Entry<Connections, Matrix> e : input.entrySet()) {
	    Connections graph = e.getKey();
	    Matrix inputMatrix = e.getValue();

	    if (inputMatrix.getColumns() != outputMatrix.getColumns() || outputMatrix.getRows() != graph.getConnectionGraph().getRows()) {
		throw new IllegalArgumentException("matrices do not match");
	    }
	}
    }

    /**
     * @param row
     *            , column - index within the output array
     */
    protected void before(int row, int column) {
    }

    /**
     * @param row
     *            , column - index within the output array
     */
    protected void after(int row, int column) {
    }
}
