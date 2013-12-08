package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * Base interface for layer calculator. The difference with the feedforward layer calculator is the "activations" parameter, which contains the activations from the feedforward phse
 */
public interface BackPropagationLayerCalculator {
    public void backpropagate(Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results, Layer layer);
}
