package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLUByRows.AparapiReLUByColumns;

public class AparapiReLUConnectionCalculator extends ConnectionCalculatorBase {

    private AparapiReLUByRows byRows;
    private AparapiReLUByColumns byColumns;

    @Override
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer) {
	super.calculate(connection, input, output, targetLayer);

	if (connection.getOutputLayer() == targetLayer) {
	    if (byRows == null) {
		byRows = new AparapiReLUByRows();
	    }

	    byRows.calculate(connection, input, output);
	} else {
	    if (byColumns == null) {
		byColumns = new AparapiReLUByColumns();
	    }
	    
	    byColumns.calculate(connection, input, output);
	}
    }
}
