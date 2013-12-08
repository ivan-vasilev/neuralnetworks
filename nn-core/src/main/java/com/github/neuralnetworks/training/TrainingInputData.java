package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Training input data with target value
 * Batch Input and target are provided as matrices (each column/row is one training example)
 */
public interface TrainingInputData {
    public Matrix getInput();
    public Matrix getTarget();
}
