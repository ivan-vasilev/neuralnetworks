package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Rectified linear unit convolutional calculator
 */
public class AparapiConv2DReLU extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    @Override
    protected AparapiConv2D createInputFunction(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiConv2DReLUFunction(c, valuesProvider, targetLayer);
    }

    public static class AparapiConv2DReLUFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	public AparapiConv2DReLUFunction(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(c, valuesProvider, targetLayer);
	}

	@Override
	protected float activationFunction(float value) {
	    return max(0, value);
	}
    }
}
