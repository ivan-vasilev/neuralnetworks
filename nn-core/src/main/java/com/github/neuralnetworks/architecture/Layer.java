package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * A layer of neurons. Each layer contains a set of connections that link it to other layers.
 * In that sense every neural network is a graph. This is done for maximum versatility.
 * It makes possible the representation of various architectures - committee of machines or parallel networks to be calculated on different GPU devices.
 */
public class Layer implements Serializable
{

	private String name = null;
	private int[] layerDimension = new int[0];

	private static final long serialVersionUID = 1035633207383317489L;

	/**
	 * Set of links to other layers
	 */
	private List<Connections> connections;

	public Layer()
	{
		this(null);
	}

	public Layer(String name)
	{
		super();
        this.name=name;
		this.connections = new UniqueList<>();
	}

	/**
	 * @param network
	 * @return list of connections within the specific neural network
	 */
	public List<Connections> getConnections(NeuralNetwork network)
	{
		return connections.stream().filter(c -> network.getLayers().contains(Util.getOppositeLayer(c, this))).collect(Collectors.toList());
	}

	public List<Connections> getConnections()
	{
		return connections;
	}

	public void setConnections(List<Connections> connections)
	{
		this.connections = connections;
	}

	public void addConnection(Connections connection)
	{
		if (connections == null)
		{
			connections = new UniqueList<>();
		}

		connections.add(connection);
	}

	public int getUnitCount(Collection<Connections> connections)
	{
		int result = 0;
		for (Connections c : connections)
		{
			if (c.getInputLayer() == this)
			{
				if (result == 0)
				{
					result = c.getInputUnitCount();
				}

				if (result != c.getInputUnitCount())
				{
					throw new IllegalArgumentException("Some connections require different unit count");
				}
			} else if (c.getOutputLayer() == this)
			{
				if (result == 0)
				{
					result = c.getOutputUnitCount();
				}

				if (result != c.getOutputUnitCount())
				{
					throw new IllegalArgumentException("Some connections require different unit count");
				}
			} else
			{
				throw new IllegalArgumentException("A connection doesn't have the targetLayer as either input or output");
			}
		}

		return result;
	}

	public String getName()
	{
		return name;
	}

	public void setName(String name)
	{
		this.name = name;
	}

	public int[] getLayerDimension()
	{
		return layerDimension;
	}

	public void setLayerDimension(int[] layerDimension)
	{
		if (layerDimension == null)
		{
			throw new IllegalArgumentException("layerDimension must be not null!");
		}

		this.layerDimension = layerDimension;
	}

	public int getNeuronCount()
	{
		int count = 0;
		boolean first = true;
		for (int d : layerDimension)
		{
			if (first)
			{
				count = d;
				first = false;
			} else
			{
				count = count * d;
			}
		}
		return count;
	}

	public Pair<List<Layer>,List<Layer>> getInputAndOutputLayer()
    {
        
        List<Layer> listOfInputLayer=new ArrayList<>();
        List<Layer> listOfOutputLayer=new ArrayList<>();
        
        for (Connections connection : connections) {
            if(connection.getOutputLayer()==this)
            {
                listOfInputLayer.add(connection.getInputLayer());
            }
            if(connection.getInputLayer()==this)
            {
                listOfOutputLayer.add(connection.getOutputLayer());
            }
        }
        
        return new Pair<List<Layer>,List<Layer>>(listOfInputLayer,listOfOutputLayer);
    }

	public Pair<List<Connections>,List<Connections>> getInputAndOutputConnection()
    {

        List<Connections> listOfInputConnections=new ArrayList<>();
        List<Connections> listOfOutputConnections=new ArrayList<>();

        for (Connections connection : connections) {
            if(connection.getOutputLayer()==this)
            {
                listOfInputConnections.add(connection);
            }
            if(connection.getInputLayer()==this)
            {
                listOfOutputConnections.add(connection);
            }
        }

        return new Pair<List<Connections>,List<Connections>>(listOfInputConnections,listOfOutputConnections);
    }

	@Override
	public String toString()
	{
		StringBuilder builder = new StringBuilder();

		builder.append(name).append("\n");
		if (this.getLayerDimension() != null)
		{
			builder.append("neurons: ").append(this.getNeuronCount()).append("\n");
		}

		builder.append("\n");
		Pair<List<Layer>, List<Layer>> inputAndOutputLayer = getInputAndOutputLayer();
		builder.append("input:\n");
		for (Layer layer : inputAndOutputLayer.getLeft())
		{
			builder.append(layer.getName()).append("\n");
		}

		builder.append("\n");
		builder.append("output:\n");
		for (Layer layer : inputAndOutputLayer.getRight())
		{
			builder.append(layer.getName()).append("\n");
		}

		return builder.toString();
	}
}
