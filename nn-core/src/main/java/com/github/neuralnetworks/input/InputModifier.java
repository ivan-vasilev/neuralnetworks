package com.github.neuralnetworks.input;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * implementations of this interface are used for modifying the input
 *
 */
public interface InputModifier {
    public Matrix modify(Matrix input);
}
