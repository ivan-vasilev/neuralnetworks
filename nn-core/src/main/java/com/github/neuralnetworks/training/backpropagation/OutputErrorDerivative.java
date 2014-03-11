package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations provide output error derivative
 */
public interface OutputErrorDerivative extends Serializable {
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target);
}
