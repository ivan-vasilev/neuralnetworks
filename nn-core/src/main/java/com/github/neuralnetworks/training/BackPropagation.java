package com.github.neuralnetworks.training;

import java.util.Map;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

public interface BackPropagation {
    public void backPropagate(Map<Layer, Matrix> activations, Matrix outputError, NeuralNetwork layers);
    public Matrix getOutputErrorDerivative(Matrix activation, Matrix target);
}
