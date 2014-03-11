package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations provide transformations to the elements of the matrix
 */
public interface MatrixFunction extends Serializable {
    public void value(Matrix inputOutput);
}
