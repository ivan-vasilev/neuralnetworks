package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;

/**
 * Softplus convolutional calculator
 */
public class AparapiConv2DSoftReLU extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    @Override
    protected AparapiConv2D createInputFunction(Conv2DConnection c, int miniBatchSize, Layer targetLayer) {
	return new AparapiConv2DSoftReLUFunction(c, miniBatchSize);
    }

    public static class AparapiConv2DSoftReLUFunction extends AparapiConv2DFF {

	private static final long serialVersionUID = -7985734201416578973L;

	public AparapiConv2DSoftReLUFunction(Conv2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected float activationFunction(float value) {
	    return log(1 + exp(value));
	}
    }
}
