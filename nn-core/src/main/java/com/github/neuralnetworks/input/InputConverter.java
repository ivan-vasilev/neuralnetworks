package com.github.neuralnetworks.input;

import java.io.Serializable;

/**
 * Converts input values to matrices
 */
public interface InputConverter extends Serializable {
    public void convert(Object input, float[] output);
}
