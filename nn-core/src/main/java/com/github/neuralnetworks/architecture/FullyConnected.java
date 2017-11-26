package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.TensorFactory;

/**
 * Fully connected weight matrix between two layers of neurons
 */
public class FullyConnected extends ConnectionsImpl implements WeightsConnections
{

	private static final long serialVersionUID = 1L;

	/**
	 * Weight matrix for the weights of the links
	 */
	private Matrix connectionGraph;

	public FullyConnected(Layer inputLayer, Layer outputLayer, int inputUnitCount, int outputUnitCount)
	{
		this(inputLayer, outputLayer, TensorFactory.matrix(new float[inputUnitCount * outputUnitCount], inputUnitCount));
	}

	public FullyConnected(Layer inputLayer, Layer outputLayer, Matrix connectionGraph)
	{
		super(inputLayer, outputLayer);
		this.connectionGraph = connectionGraph;
	}

	@Override
	public Matrix getWeights()
	{
		return connectionGraph;
	}

	public void setWeights(Matrix weights)
	{
		this.connectionGraph = weights;
	}

	@Override
	public int getInputUnitCount()
	{
		return connectionGraph.getColumns();
	}

	@Override
	public int getOutputUnitCount()
	{
		return connectionGraph.getRows();
	}
}
