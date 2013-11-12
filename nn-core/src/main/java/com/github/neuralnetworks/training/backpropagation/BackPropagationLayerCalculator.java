package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;

public interface BackPropagationLayerCalculator {
    public void backpropagate(Set<Layer> calculatedLayers, Map<Layer, Matrix> activations, Map<Layer, Matrix> results, Layer layer);
}
