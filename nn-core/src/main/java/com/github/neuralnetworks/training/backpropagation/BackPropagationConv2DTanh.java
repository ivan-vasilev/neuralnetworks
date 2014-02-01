package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Tanh Convolutional Backpropagation connection calculator for relu units
 */
public class BackPropagationConv2DTanh extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DTanh(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackpropagationConnectionCalculator> connectionCalculators, Layer targetLayer) {
	Conv2DConnection con = null;
	for (Connections c : inputConnections.keySet()) {
	    if (c instanceof Conv2DConnection && !Util.isBias(c.getInputLayer())) {
		con = (Conv2DConnection) c;
		break;
	    }
	}

	if (con != null) {
	    connectionCalculators.put(con, new AparapiBackpropConv2DTanh(con, miniBatchSize));
	}
    }

    public static class AparapiBackpropConv2DTanh extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropConv2DTanh(Conv2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected float activationFunctionDerivative(float value) {
	    return 1 - value * value;
	}
    }
}
