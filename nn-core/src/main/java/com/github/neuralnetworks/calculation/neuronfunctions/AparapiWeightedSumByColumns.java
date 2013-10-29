package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

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

	    for (int k = 0; k < series; k++) {
		for (int j = 0; j < weightsRows[0]; j++) {
		    output[outputIndex(id, i, k)] += input[inputIndex(j, i, k)] * weights[weightIndex(j, id, k)];
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
