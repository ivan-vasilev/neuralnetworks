package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * Weighted sum by columns (it is used in the backpropagation case)
 */
public class AparapiWeightedSumByColumns extends AparapiBaseFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected int[] weightsRows;

    @Override
    public void run() {
	int id = getGlobalId();

	// each input example
	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    // each connection (of the combined connections)
	    for (int k = 0; k < series; k++) {

		// each element in the column
		float value = 0;
		for (int j = 0; j < weightsRows[k]; j++) {
		    value += input[inputIndex(j, i, k)] * weights[weightIndex(j, id, k)];
		}
		output[outputIndex(id, i, k)] = value;
	    }

	    after(id, i);
	}
    }

    /**
     * initialization before the actual calculation
     */
    @Override
    protected void init(SortedMap<GraphConnections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	super.init(input, outputMatrix, targetLayer);

	weightsRows = new int[series];

	int i = 0;
	for (java.util.Map.Entry<GraphConnections, Matrix> e : input.entrySet()) {
	    GraphConnections graph = e.getKey();
	    Matrix cg = graph.getConnectionGraph();
	    Matrix inputMatrix = e.getValue();

	    if (inputMatrix.getColumns() != outputMatrix.getColumns() || (outputMatrix.getRows() != cg.getColumns() && cg.getColumns() != 1)) {
		throw new IllegalArgumentException("matrices do not match");
	    }

	    weightsRows[i++] = cg.getRows();
	}
    }

    protected void before(int row, int column) {
    }

    protected void after(int row, int column) {
    }
}
