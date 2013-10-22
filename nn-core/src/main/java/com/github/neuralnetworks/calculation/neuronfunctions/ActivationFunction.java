package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * this is a transfer function for a layer of neurons
 * 
 */
public interface ActivationFunction {
    public void value(Matrix inputOutput);
}
