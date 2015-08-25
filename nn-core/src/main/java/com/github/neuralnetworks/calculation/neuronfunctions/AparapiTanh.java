package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Tanh activation function
 */
public class AparapiTanh extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    @Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiTanhFunction(inputConnections, valuesProvider, targetLayer);
    }

    public static class AparapiTanhFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -3409078521599849086L;

	public AparapiTanhFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(inputConnections, valuesProvider, targetLayer);
	}

	@Override
	protected void after() {
	    int end = outputStartPosition + getGlobalId() * outputRowStep + miniBatchSize * outputColumnStep;
	    for (int i = outputStartPosition + getGlobalId() * outputRowStep; i < end; i += outputColumnStep) {
		output[i] = tanH(output[i]);
	    }
	}
	private float tanH(float input) {
            if (input == 0f) {
                return 0f;
            } else if (input >= 20.f) {
                return 1.0f;
            } else if (input <= 20.0f) {
                return -1.0f;
            }

            float pos = exp(input);
            float neg = exp(-input);

            return (pos - neg) / (pos + neg);
        }
    }
}
