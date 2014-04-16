package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;

import com.github.neuralnetworks.util.Tensor;

/**
 * Implementations provide transformations to the elements of the matrix
 */
public interface TensorFunction extends Serializable {
    public void value(Tensor inputOutput);
}
