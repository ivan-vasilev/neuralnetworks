package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations provide output error derivative
 */
public interface OutputErrorDerivative {
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target);
}
