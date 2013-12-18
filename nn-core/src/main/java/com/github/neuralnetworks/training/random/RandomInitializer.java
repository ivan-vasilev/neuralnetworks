package com.github.neuralnetworks.training.random;

import java.io.Serializable;

/**
 * Base interface for random initialization of arrays
 */
public interface RandomInitializer extends Serializable {
    public void initialize(float[] array);
}
