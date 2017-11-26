package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * Base Aparapi connection calculator for fully connected layers.
 * If there are multiple inbound connections they are combined
 * in a "single" connection and are calculated simultaneously
 * 
 * !!! IMPORTANT !!! Aparapi only works one-dimensional arrays of primitive data
 * types can only call member methods of the Kernel class itself.
 * 
 * Because of this limitations all the data that is contained in the input
 * connections, weight matrices, input values etc is converted into
 * one-dimensional member arrays of this class
 */
public abstract class AparapiFullyConnected extends Kernel implements ConnectionCalculator
{

	private static final long serialVersionUID = -8435155322138790083L;

	/**
	 * Number of input samples that will be calculated simultaneously
	 */
	protected final int miniBatchSize;

	/**
	 * This is combined with the other properties to represent the
	 * FullyConnected connection (the FullyConnected class itself cannot be used
	 * because of the Aparapi limitations) It is an array, because of the
	 * combined connections
	 */
	protected float[] input;
	@Constant
	protected final int inputStartPosition;
	@Constant
	protected final int inputRowStep;
	@Constant
	protected final int inputColumnStep;

	/**
	 * output values
	 */
	protected float[] output;
	protected final int outputStartPosition;
	protected final int outputRowStep;
	protected final int outputColumnStep;

	protected final float[] weights;
	@Constant
	protected final int weightStartPosition;
	@Constant
	protected final int weightsSize;
	@Constant
	protected final int weightsInitialStep;
	@Constant
	protected final int weightsStep;

	public AparapiFullyConnected(Connections inputConnection, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super();

		if (!(inputConnection instanceof FullyConnected))
		{
			throw new IllegalArgumentException("Only FullyConnected connections are supported");
		}

		this.miniBatchSize = TensorFactory.batchSize(valuesProvider);

		// input
		input = TensorFactory.tensor(Util.getOppositeLayer(inputConnection, targetLayer), inputConnection, valuesProvider).getElements();

		Tensor t = TensorFactory.tensor(Util.getOppositeLayer(inputConnection, targetLayer), inputConnection, valuesProvider);
		if (!(t instanceof Matrix))
		{
			throw new IllegalArgumentException("Only matrices are supported as input");
		}

		Matrix m = TensorFactory.tensor(Util.getOppositeLayer(inputConnection, targetLayer), inputConnection, valuesProvider);
		this.inputStartPosition = m.getStartIndex();
		this.inputRowStep = m.getRowElementsDistance();
		this.inputColumnStep = m.getColumnElementsDistance();

		// output
		Matrix o = TensorFactory.tensor(targetLayer, inputConnection, valuesProvider);
		this.output = o.getElements();
		this.outputStartPosition = o.getStartIndex();
		this.outputRowStep = o.getRowElementsDistance();
		this.outputColumnStep = o.getColumnElementsDistance();

		// weights
		weights = ((FullyConnected) inputConnection).getWeights().getElements();
		Matrix w = ((FullyConnected) inputConnection).getWeights();
		weightStartPosition = w.getStartIndex();
		if (inputConnection.getOutputLayer() == targetLayer)
		{
			weightsSize = w.getColumns();
			weightsInitialStep = w.getRowElementsDistance();
			weightsStep = w.getColumnElementsDistance();
		} else
		{
			weightsSize = w.getRows();
			weightsInitialStep = w.getColumnElementsDistance();
			weightsStep = w.getRowElementsDistance();
		}
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (accept(connections, valuesProvider, targetLayer))
		{
			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
		} else
		{
			throw new IllegalArgumentException("A parameter does not match");
		}
	}

	public boolean accept(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (connections.size() != 1)
		{
			return false;
		}

		if (TensorFactory.batchSize(valuesProvider) != miniBatchSize)
		{
			return false;
		}

		if (TensorFactory.tensor(targetLayer, connections, valuesProvider).getElements() != output)
		{
			return false;
		}

		if (TensorFactory.tensor(Util.getOppositeLayer(connections.get(0), targetLayer), connections.get(0), valuesProvider).getElements() != input)
		{
			return false;
		}

		return true;
	}

	public float[] getInput()
	{
		return input;
	}

	public void setInput(float[] input)
	{
		this.input = input;
	}

	public float[] getOutput()
	{
		return output;
	}

	public void setOutput(float[] output)
	{
		this.output = output;
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public int getInputStartPosition()
	{
		return inputStartPosition;
	}

	public int getInputRowStep()
	{
		return inputRowStep;
	}

	public int getInputColumnStep()
	{
		return inputColumnStep;
	}

	public int getOutputStartPosition()
	{
		return outputStartPosition;
	}

	public int getOutputRowStep()
	{
		return outputRowStep;
	}

	public int getOutputColumnStep()
	{
		return outputColumnStep;
	}

	public float[] getWeights()
	{
		return weights;
	}

	public int getWeightStartPosition()
	{
		return weightStartPosition;
	}

	public int getWeightsSize()
	{
		return weightsSize;
	}

	public int getWeightsInitialStep()
	{
		return weightsInitialStep;
	}

	public int getWeightsStep()
	{
		return weightsStep;
	}
}
