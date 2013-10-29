package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

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

	    for (int k = 0; k < series; k++) {
		for (int j = 0; j < weightsColumns[0]; j++) {
		    output[outputIndex(id, i, k)] += input[inputIndex(j, i, k)] * weights[weightIndex(id, j, k)];
		}
	    }

	    after(id, i);
	}
    }

    /**
     * initialization before the actual calculation
     */
    @Override
    protected void init(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
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
