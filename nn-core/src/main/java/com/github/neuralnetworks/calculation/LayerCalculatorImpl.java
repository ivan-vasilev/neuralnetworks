package com.github.neuralnetworks.calculation;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;

/**
 * Implementation of the LayerCalculator interface for calculating single target layer
 * It takes advantage of the fact that the neural network is a graph with layers as nodes and connections between layers as links of the graph
 * The results are propagated within the graph
 */
public class LayerCalculatorImpl extends LayerCalculatorBase implements LayerCalculator
{

	private static final long serialVersionUID = 1L;

	@Override
	public void calculate(NeuralNetwork neuralNetwork, Layer layer, Set<Layer> calculatedLayers, ValuesProvider results)
	{
		neuralNetwork.getLayers().stream().filter(l -> getConnectionCalculator(l) instanceof ConstantConnectionCalculator).forEach(l -> {
			calculatedLayers.add(l);
			Tensor t = results.get(l);
			ConstantConnectionCalculator cc = (ConstantConnectionCalculator) getConnectionCalculator(l);
			if (t.getElements()[t.getStartIndex()] != cc.getValue())
			{
				t.forEach(i -> t.getElements()[i] = cc.getValue());
			}
		});

		List<ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(neuralNetwork, layer, calculatedLayers).order();

		Collections.reverse(ccc);

		calculate(results, ccc, neuralNetwork);
	}
}
