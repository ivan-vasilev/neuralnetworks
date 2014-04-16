package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

import com.github.neuralnetworks.util.Tensor;

/**
 * Implementations provide output error derivative
 */
public interface OutputErrorDerivative extends Serializable {
    public void getOutputErrorDerivative(Tensor activation, Tensor target, Tensor result);
}
