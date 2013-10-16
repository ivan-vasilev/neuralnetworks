package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * training input data with target value
 * 
 */
public interface TrainingInputData {
    public Matrix getInput();
    public Matrix getTarget();
}
