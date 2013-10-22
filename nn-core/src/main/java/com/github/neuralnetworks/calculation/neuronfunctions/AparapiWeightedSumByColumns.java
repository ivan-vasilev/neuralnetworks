package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.OneToOne;

public class AparapiWeightedSumByColumns extends AparapiBaseFunction implements InputFunction {

    private static final long serialVersionUID = 8288998425211708411L;

    protected int weightsRows;

    @Override
    public void run() {
	int id = getGlobalId();

	for (int i = 0; i < inputOutputColumns; i++) {
	    before(id, i);

	    for (int j = 0; j < weightsRows; j++) {
		output[outputIndex(id, i)] += input[inputIndex(j, i)] * weights[weightIndex(j, id)];
	    }

	    after(id, i);
	}
    }

    /**
     * initialization before the actual calculation
     */
    @Override
    protected void init(Connections graph, Matrix inputMatrix, Matrix outputMatrix) {
	super.init(graph, inputMatrix, outputMatrix);

	Matrix cg = graph.getConnectionGraph();

	if (inputMatrix.getColumns() != outputMatrix.getColumns() || (outputMatrix.getRows() != cg.getColumns() && cg.getColumns() != 1)) {
	    throw new IllegalArgumentException("matrices do not match");
	}

	this.weightsRows = graph instanceof OneToOne ? 1 : cg.getRows();
    }
    
    protected void before(int row, int column) {
    }

    protected void after(int row, int column) {
    }
}
