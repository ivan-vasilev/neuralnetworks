package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoidByRows.AparapiSigmoidByColumns;

public class AparapiSigmoidConnectionCalculator implements ConnectionCalculator {

    private AparapiSigmoidByRows byRows;
    private AparapiSigmoidByColumns byColumns;

    @Override
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer) {
	if (connection.getOutputLayer() == targetLayer) {
	    if (byRows == null) {
		byRows = new AparapiSigmoidByRows();
	    }

	    byRows.calculate(connection, input, output);
	} else {
	    if (byColumns == null) {
		byColumns = new AparapiSigmoidByColumns();
	    }
	    
	    byColumns.calculate(connection, input, output);
	}
    }
}
