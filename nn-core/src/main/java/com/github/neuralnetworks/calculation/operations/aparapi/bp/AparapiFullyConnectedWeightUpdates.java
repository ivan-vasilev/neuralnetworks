package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.util.Environment;

/**
 * Aparapi Backpropagation base weighted sum Supports learning rate, momentum
 * and weight decay
 */
public class AparapiFullyConnectedWeightUpdates extends Kernel implements WeightUpdates
{
	private static final long serialVersionUID = -5101971690861270462L;

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
	protected Tensor inputTensor;
	protected float[] input;
	@Constant
	protected final int inputStartPosition;
	@Constant
	protected final int inputRowStep;
	@Constant
	protected final int inputColumnStep;

	/**
	 * Activation of the output layer from the feedforward phase
	 */
	@Constant
	protected float[] ffActivation;
	protected final int activationStartPosition;
	protected final int activationRowStep;
	protected final int activationColumnStep;

	/**
	 * weights
	 */
	protected final Tensor weightsTensor;
	protected final float[] weights;
	@Constant
	protected final int weightStartPosition;
	@Constant
	protected final int weightsRows;
	@Constant
	protected final int weightsRowsDistance;
	@Constant
	protected final int weightsColumns;
	@Constant
	protected final int weightsColumnsDistance;

	/**
	 * Weight updates array
	 */
	protected final float[] weightUpdates;

	protected float learningRate;
	protected float momentum;
	protected float l1weightDecay;
	protected float l2weightDecay;

	public AparapiFullyConnectedWeightUpdates(Connections connection, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
	{
		if (!(connection instanceof FullyConnected))
		{
			throw new IllegalArgumentException("Only FullyConnected connections are supported");
		}

		this.miniBatchSize = TensorFactory.batchSize(valuesProvider);

		Matrix activation = TensorFactory.tensor(connection.getInputLayer(), connection, activations);
		this.ffActivation = activation.getElements();
		this.activationStartPosition = activation.getStartIndex();
		this.activationRowStep = activation.getRowElementsDistance();
		this.activationColumnStep = activation.getColumnElementsDistance();

		Matrix in = TensorFactory.tensor(connection.getOutputLayer(), connection, valuesProvider);
		this.inputTensor = in;
		this.input = in.getElements();
		this.inputStartPosition = in.getStartIndex();
		this.inputRowStep = in.getRowElementsDistance();
		this.inputColumnStep = in.getColumnElementsDistance();

		// weights
		Matrix w = ((FullyConnected) connection).getWeights();
		this.weightsTensor = w;
		this.weights = w.getElements();
		this.weightStartPosition = w.getStartIndex();
		this.weightsRows = w.getRows();
		this.weightsRowsDistance = w.getRowElementsDistance();
		this.weightsColumns = w.getColumns();
		this.weightsColumnsDistance = w.getColumnElementsDistance();

		this.weightUpdates = weightUpdates.getElements();
	}

	@Override
	public void updateWeights(float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		this.learningRate = learningRate;
		this.momentum = momentum;
		this.l1weightDecay = l1weightDecay;
		this.l2weightDecay = l2weightDecay;

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, weightsTensor.getSize());
	}

	@Override
	public void run()
	{
		int id = getGlobalId();

		int row = id / weightsColumns;
		int column = id % weightsColumns;

		float weightUpdate = 0;
		for (int i = 0; i < miniBatchSize; i++)
		{
			weightUpdate += input[inputStartPosition + i * inputRowStep + row * inputColumnStep] * ffActivation[activationStartPosition + i * activationRowStep + column * activationColumnStep];
		}

		int weightIndex = weightStartPosition + row * weightsRowsDistance + column * weightsColumnsDistance;
		float weight = weights[weightIndex];
		weightUpdate = (learningRate * weightUpdate) / miniBatchSize + momentum * weightUpdates[weightIndex] - learningRate * l1weightDecay * weight - l2weightDecay * weight * weight;
		weights[weightIndex] += weightUpdate;
		weightUpdates[weightIndex] = weightUpdate;
	}

	public float getLearningRate()
	{
		return learningRate;
	}

	public void setLearningRate(float learningRate)
	{
		this.learningRate = learningRate;
	}

	public float getMomentum()
	{
		return momentum;
	}

	public void setMomentum(float momentum)
	{
		this.momentum = momentum;
	}

	public float getL1weightDecay()
	{
		return l1weightDecay;
	}

	public void setL1weightDecay(float l1weightDecay)
	{
		this.l1weightDecay = l1weightDecay;
	}

	public float getL2weightDecay()
	{
		return l2weightDecay;
	}

	public void setL2weightDecay(float l2weightDecay)
	{
		this.l2weightDecay = l2weightDecay;
	}

	public float[] getFfActivation()
	{
		return ffActivation;
	}

	public int getActivationStartPosition()
	{
		return activationStartPosition;
	}

	public int getActivationRowStep()
	{
		return activationRowStep;
	}

	public int getActivationColumnStep()
	{
		return activationColumnStep;
	}

	public float[] getWeightUpdates()
	{
		return weightUpdates;
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public float[] getInput()
	{
		return input;
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

	public Tensor getWeightsTensor()
	{
		return weightsTensor;
	}

	public float[] getWeights()
	{
		return weights;
	}

	public int getWeightStartPosition()
	{
		return weightStartPosition;
	}

	public int getWeightsRows()
	{
		return weightsRows;
	}

	public int getWeightsRowsDistance()
	{
		return weightsRowsDistance;
	}

	public int getWeightsColumns()
	{
		return weightsColumns;
	}

	public int getWeightsColumnsDistance()
	{
		return weightsColumnsDistance;
	}

	public Tensor getInputTensor()
	{
		return inputTensor;
	}
}
