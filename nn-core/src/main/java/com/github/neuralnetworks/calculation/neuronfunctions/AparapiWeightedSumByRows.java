package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * Weighted sum by rows (it is used in the feedforward case)
 */
public class AparapiWeightedSumByRows extends AparapiBaseFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    @Override
    public void run() {
	int id = getGlobalId();

	// each input example
	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    // each connection (of the combined connections)
	    for (int k = 0; k < series; k++) {
		// each element in the row
		float value = 0;

		for (int j = 0; j < weightsColumns[k]; j++) {
		    value += input[inputIndex(j, i, k)] * weights[weightIndex(id, j, k)];
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

	for (java.util.Map.Entry<GraphConnections, Matrix> e : input.entrySet()) {
	    GraphConnections graph = e.getKey();
	    Matrix inputMatrix = e.getValue();

	    if (inputMatrix.getColumns() != outputMatrix.getColumns() || outputMatrix.getRows() != graph.getConnectionGraph().getRows()) {
		throw new IllegalArgumentException("matrices do not match");
	    }
	}
    }

    /**
     * Executed before the sum
     * @param row, column - index within the output array
     */
    protected void before(int row, int column) {
    }

    /**
     * Executed after the sum
     * @param row, column - index within the output array
     */
    protected void after(int row, int column) {
    }
}
