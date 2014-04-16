package com.github.neuralnetworks.training.random;

import java.io.Serializable;

import com.github.neuralnetworks.util.Tensor;

/**
 * Base interface for random initialization of arrays
 */
public interface RandomInitializer extends Serializable {
    public void initialize(Tensor t);
}
