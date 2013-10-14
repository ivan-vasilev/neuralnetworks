package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.InputOutputLayers;
import com.github.neuralnetworks.architecture.Matrix;

public interface BackPropagation {
    public void backPropagate(Matrix outputError, InputOutputLayers layers);
    public Matrix getOutputErrorDerivative(Matrix actual, Matrix target);
}
