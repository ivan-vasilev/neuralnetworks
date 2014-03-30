package com.github.neuralnetworks.training.backpropagation;

import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Convolutional Backpropagation connection calculator for relu units
 */
public class BackPropagationConv2DReLU extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2DReLU(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	Conv2DConnection con = null;
	for (Connections c : inputConnections) {
	    if (c instanceof Conv2DConnection && !Util.isBias(c.getInputLayer())) {
		con = (Conv2DConnection) c;
		break;
	    }
	}

	if (con != null) {
	    connectionCalculators.put(con, new AparapiBackpropConv2DReLU(con, valuesProvider, targetLayer));
	}
    }

    public static class AparapiBackpropConv2DReLU extends AparapiBackpropagationConv2D {

	private static final long serialVersionUID = -3580345016542506932L;

	public AparapiBackpropConv2DReLU(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(c, valuesProvider, targetLayer);
	}

	@Override
	protected float activationFunctionDerivative(float value) {
	    if (value > 0) {
		return value;
	    } else {
		return 0;
	    }
	}
    }
}
