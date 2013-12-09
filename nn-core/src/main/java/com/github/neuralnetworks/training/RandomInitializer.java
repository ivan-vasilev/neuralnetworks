package com.github.neuralnetworks.training;

import java.util.Random;

/**
 * Base interface for random initialization of arrays
 */
public interface RandomInitializer {
    public void initialize(float[] array);
    public Random getRandom();
}
