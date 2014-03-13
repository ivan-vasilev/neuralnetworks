package com.github.neuralnetworks.input;

import java.io.Serializable;

import com.github.neuralnetworks.util.Matrix;

/**
 * Converts input values to matrices
 */
public interface InputConverter extends Serializable {
    public abstract Matrix convert(Object[] input);
}
