package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator without activation function (for connections to the output layer)
 */
public class BackPropagationConv2D extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1178188233641224762L;

    public BackPropagationConv2D(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(SortedMap<Connections, Integer> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, Layer targetLayer) {
	Conv2DConnection con = null;
	for (Connections c : inputConnections.keySet()) {
	    if (c instanceof Conv2DConnection) {
		con = (Conv2DConnection) c;
		break;
	    }
	}

	if (con != null) {
	    connectionCalculators.put(con, new AparapiBackpropagationConv2D(con, valuesProvider, targetLayer));
	}
    }
}
