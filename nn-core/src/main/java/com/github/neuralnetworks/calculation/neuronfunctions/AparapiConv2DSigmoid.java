package com.github.neuralnetworks.calculation.neuronfunctions;

public class AparapiConv2DSigmoid extends AparapiConv2D {

    private static final long serialVersionUID = -7985734201416578973L;

    @Override
    protected void after() {
	output[getGlobalId()] = 1 / (1 + exp(-output[getGlobalId()]));
    }
}
