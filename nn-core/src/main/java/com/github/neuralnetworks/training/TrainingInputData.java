package com.github.neuralnetworks.training;

import java.io.Serializable;

import com.github.neuralnetworks.util.Matrix;

/**
 * Training input data with target value
 * Batch Input and target are provided as matrices (each column/row is one training example)
 */
public interface TrainingInputData extends Serializable {
    public Matrix getInput();
    public Matrix getTarget();
}
