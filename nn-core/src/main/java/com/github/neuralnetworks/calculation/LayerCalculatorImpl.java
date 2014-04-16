package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Implementation of the LayerCalculator interface for calculating single target layer
 * It takes advantage of the fact that the neural network is a graph with layers as nodes and connections between layers as links of the graph
 * The results are propagated within the graph
 */
public class LayerCalculatorImpl extends LayerCalculatorBase implements LayerCalculator, Serializable {

    private static final long serialVersionUID = 1L;

    @Override
    public void calculate(NeuralNetwork neuralNetwork, Layer layer, Set<Layer> calculatedLayers, ValuesProvider results) {
	List<ConnectionCandidate> ccc = new TargetLayerOrderStrategy(neuralNetwork, layer, calculatedLayers).order();
	calculate(results, ccc, neuralNetwork);
    }
}
