package com.github.neuralnetworks.input;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Converts input values to matrices
 */
public interface InputConverter {
    public abstract Matrix convert(Object[] input);
}
