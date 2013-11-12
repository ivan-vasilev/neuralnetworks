package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;

public interface OutputErrorDerivative {
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target);
}
