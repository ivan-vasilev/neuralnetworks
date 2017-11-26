package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiWeightedSum;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;

/**
 * Aparapi Backpropagation base weighted sum Supports learning rate, momentum
 * and weight decay
 */
public class AparapiBackpropagationFullyConnected extends AparapiWeightedSum implements BackPropagationConnectionCalculator
{

	private static final long serialVersionUID = -5101971690861270462L;

	/**
	 * Activation of the output layer from the feedforward phase
	 */
	@Constant
	protected float[] ffActivation;
	protected int activationStartPosition;
	protected int activationRowStep;
	protected int activationColumnStep;

	/**
	 * Weight updates array
	 */
	protected float[] weightUpdates;

	protected float learningRate;
	protected float momentum;
	protected float l1weightDecay;
	protected float l2weightDecay;

	protected AparapiFullyConnectedWeightUpdates aparapiWeightUpdates;

	@SuppressWarnings("unused")
	public AparapiBackpropagationFullyConnected(Connections inputConnection, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates, Layer targetLayer)
	{
		super(inputConnection, valuesProvider, targetLayer);

//		Matrix m = TensorFactory.tensor(targetLayer, inputConnection, activations);
//		this.ffActivation = m.getElements();
//		this.activationStartPosition = m.getStartIndex();
//		this.activationRowStep = m.getRowElementsDistance();
//		this.activationColumnStep = m.getColumnElementsDistance();
//
//		this.learningRate = momentum;
//		this.weightUpdates = weightUpdates.getElements();
//
//		this.aparapiWeightUpdates = new AparapiFullyConnectedWeightUpdates(inputConnection, valuesProvider, activations, weightUpdates);
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super.calculate(connections, valuesProvider, targetLayer);
		//FullyConnected fc = (FullyConnected) connections.get(0);
		//Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(aparapiWeightUpdates, fc.getWeights().getSize());
	}

//	@Override
//	protected void after()
//	{
//		int id = getGlobalId();
//
//		int weightIndex = 0;
//		float weight = 0, weightUpdate = 0, lr = learningRate;
//
//		// each element in the row/column
//		for (int j = 0; j < weightsSize; j++)
//		{
//			weightUpdate = 0;
//			for (int i = 0; i < miniBatchSize; i++)
//			{
//				weightUpdate += input[inputStartPosition + i * inputRowStep + j * inputColumnStep] * ffActivation[activationStartPosition + i * activationRowStep + id * activationColumnStep];
//			}
//
//			weightIndex = weightStartPosition + weightsInitialStep * id + j * weightsStep;
//			weight = weights[weightIndex];
//			weightUpdate = lr * weightUpdate + momentum * weightUpdates[weightIndex] - l1weightDecay * abs(weight) - l2weightDecay * weight * weight;
//			weights[weightIndex] += weightUpdate;
//			weightUpdates[weightIndex] = weightUpdate;
//		}
//
//		calcDerivative();
//	}
//
//	/**
//	 * calculate derivative after weights update
//	 */
//	protected void calcDerivative()
//	{
//	}

	@Override
	public ValuesProvider getActivations()
	{
		return null;
	}

	@Override
	public void setActivations(ValuesProvider activations)
	{
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
}
