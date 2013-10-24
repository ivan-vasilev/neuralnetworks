package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * implementations of this interface calculate a single connection between layers
 *
 */
public interface ConnectionCalculator extends Serializable {
    public void calculate(Map<Connections, Matrix> connections, Matrix output, Layer targetLayer);
}
