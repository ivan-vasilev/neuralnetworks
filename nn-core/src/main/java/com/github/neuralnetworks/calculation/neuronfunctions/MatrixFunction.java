package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations provide transformations to the elements of the matrix
 */
public interface MatrixFunction {
    public void value(Matrix inputOutput);
}
