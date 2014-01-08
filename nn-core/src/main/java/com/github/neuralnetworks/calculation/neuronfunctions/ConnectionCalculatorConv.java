package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Arrays;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Default implementation of Connection calculator for convolutional/subsampling layers
 */
public class ConnectionCalculatorConv implements ConnectionCalculator {

    private static final long serialVersionUID = -5405654469496055017L;

    protected AparapiConv2D inputFunction;

    public ConnectionCalculatorConv(AparapiConv2D inputFunction) {
	super();
	this.inputFunction = inputFunction;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = null;
	Conv2DConnection bias = null;

	for (Connections con : connections.keySet()) {
	    if (con instanceof Conv2DConnection) {
		if (c.getInputLayer() instanceof BiasLayer) {
		    bias = (Conv2DConnection) con;
		} else {
		    c = (Conv2DConnection) con;
		}
	    }
	}

	calculateBias(bias, output);

	if (c != null) {
	    // currently works only as a feedforward (including bp)
	    if (targetLayer == c.getOutputLayer()) {
		inputFunction.calculate(c, connections.get(c), output);
	    } else {
		inputFunction.calculate(c, output, connections.get(c));
	    }
	}
    }

    protected void calculateBias(Conv2DConnection bias, Matrix output) {
	if (bias != null) {
	    float[] o = output.getElements();
	    ConvGridLayer l = (ConvGridLayer) bias.getOutputLayer();
	    int chunk = o.length / l.getFilters();
	    for (int i = 0; i < l.getFilters(); i++) {
		Arrays.fill(o, chunk * i, chunk * (i + 1), bias.getWeights()[i]);
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
