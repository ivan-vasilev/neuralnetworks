package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

import com.github.neuralnetworks.util.Tensor;

/**
 * Implementations provide output error derivative
 */
public interface OutputErrorDerivative extends Serializable {
    public Tensor getOutputErrorDerivative(Tensor activation, Tensor target);
}
