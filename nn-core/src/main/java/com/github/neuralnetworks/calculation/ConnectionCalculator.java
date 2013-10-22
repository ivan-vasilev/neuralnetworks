package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * implementations of this interface calculate a single connection between layers
 *
 */
public interface ConnectionCalculator {
    public void calculate(Connections connection, Matrix input, Matrix output, Layer targetLayer);
}
