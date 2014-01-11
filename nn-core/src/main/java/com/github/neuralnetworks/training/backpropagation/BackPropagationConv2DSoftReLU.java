package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator for softplus layers
 */
public class BackPropagationConv2DSoftReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DSoftReLU(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Matrix> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, int inputOutputSamples, Layer targetLayer) {
	Conv2DConnection con = null;
	for (Connections c : inputConnections.keySet()) {
	    if (c instanceof Conv2DConnection && !(c.getInputLayer() instanceof BiasLayer)) {
		con = (Conv2DConnection) c;
		break;
	    }
	}

	if (con != null) {
	    connectionCalculators.put(con, new AparapiBackpropConv2DSoftReLU(con, inputOutputSamples, targetLayer));
	}
    }

    public static class AparapiBackpropConv2DSoftReLU extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropConv2DSoftReLU(Conv2DConnection c, int inputOutputSamples, Layer targetLayer) {
	    super(c, inputOutputSamples, targetLayer);
	}

	@Override
	protected float activationFunctionDerivative(float value) {
	    return (1 / (1 + exp(-value)));
	}
    }
}
