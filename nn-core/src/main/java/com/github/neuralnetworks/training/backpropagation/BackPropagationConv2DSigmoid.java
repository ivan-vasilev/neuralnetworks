package com.github.neuralnetworks.training.backpropagation;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Util;

/**
 * Convolutional Backpropagation connection calculator for sigmoid layers
 */
public class BackPropagationConv2DSigmoid extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DSigmoid(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, ValuesProvider activations, Layer targetLayer) {
	Conv2DConnection con = null;
	for (Connections c : inputConnections) {
	    if (c instanceof Conv2DConnection && !Util.isBias(c.getInputLayer())) {
		con = (Conv2DConnection) c;
		break;
	    }
	}

	if (con != null) {
	    connectionCalculators.put(con, new AparapiBackpropConv2DSigmoid(con, valuesProvider, activations, getWeightUpdates().get(con), targetLayer));
	}
    }

    public static class AparapiBackpropConv2DSigmoid extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropConv2DSigmoid(Conv2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates, Layer targetLayer) {
	    super(c, valuesProvider, activations, weightUpdates, targetLayer);
	}

	@Override
	protected float activationFunctionDerivative(float value) {
	    return value * (1 - value);
	}
    }
}
