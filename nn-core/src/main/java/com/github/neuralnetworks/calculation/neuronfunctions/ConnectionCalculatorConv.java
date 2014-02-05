package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Util;

/**
 * Default implementation of Connection calculator for convolutional/subsampling
 * layers
 */
public class ConnectionCalculatorConv implements ConnectionCalculator {

    private static final long serialVersionUID = -5405654469496055017L;

    protected AparapiConv2D inputFunction;
    protected Layer currentLayer;
    protected int miniBatchSize;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = null;
	Conv2DConnection bias = null;

	for (Connections con : connections.keySet()) {
	    if (con instanceof Conv2DConnection) {
		if (Util.isBias(con.getInputLayer())) {
		    bias = (Conv2DConnection) con;
		} else {
		    c = (Conv2DConnection) con;
		}
	    }
	}

	if (c != null) {
	    // currently works only as a feedforward (including bp)
	    if (inputFunction == null || miniBatchSize != output.getColumns()) {
		miniBatchSize = output.getColumns();
		inputFunction = createInputFunction(c, miniBatchSize);
	    }

	    if (targetLayer == c.getOutputLayer()) {
		calculateBias(bias, output);
		inputFunction.calculate(c, connections.get(c), output);
	    } else {
		calculateBias(bias, connections.get(c));
		inputFunction.calculate(c, output, connections.get(c));
	    }
	}
    }

    protected AparapiConv2D createInputFunction(Conv2DConnection c, int miniBatchSize) {
	return new AparapiConv2DFF(c, miniBatchSize);
    }

    protected void calculateBias(Conv2DConnection bias, Matrix output) {
	if (bias != null) {
	    int fm = output.getElements().length / bias.getWeights().length;
	    for (int i = 0; i < output.getElements().length; i++) {
		output.getElements()[i] += bias.getWeights()[i / fm];
	    }
	}
    }

    public AparapiConv2D getInputFunction() {
	return inputFunction;
    }

    public void setInputFunction(AparapiConv2D inputFunction) {
	this.inputFunction = inputFunction;
    }
}
