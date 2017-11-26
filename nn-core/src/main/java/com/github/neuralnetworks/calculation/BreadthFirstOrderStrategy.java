package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Util;

/**
 * Breadth first order strategy
 */
public class BreadthFirstOrderStrategy implements LayerOrderStrategy
{

	private static final long serialVersionUID = 1L;

	private NeuralNetwork neuralNetwork;
	private Layer startLayer;
	private Set<Layer> calculatedLayers;

	public BreadthFirstOrderStrategy(NeuralNetwork neuralNetwork, Layer startLayer)
	{
		super();
		this.neuralNetwork = neuralNetwork;
		this.startLayer = startLayer;
		this.calculatedLayers = new HashSet<>();
	}

	public BreadthFirstOrderStrategy(NeuralNetwork neuralNetwork, Layer startLayer, Set<Layer> calculatedLayers)
	{
		super();
		this.neuralNetwork = neuralNetwork;
		this.startLayer = startLayer;
		this.calculatedLayers = calculatedLayers;
	}

	@Override
	public List<ConnectionCandidate> order()
	{
		List<ConnectionCandidate> result = new ArrayList<>();

		Layer currentLayer = startLayer;

		Queue<Layer> queue = new LinkedList<>();
		queue.clear();
		queue.add(startLayer);
		while (!queue.isEmpty())
		{
			currentLayer = queue.remove();
			for (Connections c : currentLayer.getConnections(neuralNetwork))
			{
				if (!result.stream().filter(cc -> cc.connection == c).findAny().isPresent() && !(calculatedLayers.contains(c.getInputLayer()) && calculatedLayers.contains(c.getOutputLayer())))
				{
					result.add(new ConnectionCandidate(c, currentLayer));
					queue.add(Util.getOppositeLayer(c, currentLayer));
				}
			}
		}

		return result;
	}

	public NeuralNetwork getNeuralNetwork()
	{
		return neuralNetwork;
	}

	public void setNeuralNetwork(NeuralNetwork neuralNetwork)
	{
		this.neuralNetwork = neuralNetwork;
	}

	public Layer getStartLayer()
	{
		return startLayer;
	}

	public void setStartLayer(Layer startLayer)
	{
		this.startLayer = startLayer;
	}

	public Set<Layer> getCalculatedLayers()
	{
		return calculatedLayers;
	}

	public void setCalculatedLayers(Set<Layer> calculatedLayers)
	{
		this.calculatedLayers = calculatedLayers;
	}
}
