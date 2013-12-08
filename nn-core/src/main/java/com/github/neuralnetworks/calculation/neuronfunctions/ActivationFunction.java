package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * A transfer function for a layer of neurons
 */
public interface ActivationFunction {
    public void value(Matrix inputOutput);
}
