package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiFullyConnected;
import com.github.neuralnetworks.calculation.operations.cpu.MaxoutWinners;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Properties;

public class BackpropagationMaxout extends BackPropagationConnectionCalculatorImpl
{

	private static final long serialVersionUID = 1L;

	public BackpropagationMaxout(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		for (Connections c : inputConnections)
		{
			connectionCalculators.put(c, new AparapiBackpropMaxout(c, valuesProvider, activations, Arrays.asList(getWeightUpdates().get(c)), 0, 0, 0,
					0));
		}
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		targetLayer = connections.get(0).getOutputLayer();
		for (Connections c : connections)
		{
			if (targetLayer != c.getOutputLayer())
			{
				throw new IllegalArgumentException("No common target layer");
			}
		}

		super.calculate(connections, valuesProvider, targetLayer);
	}

	public static class AparapiBackpropMaxout extends AparapiFullyConnected implements BackPropagationConnectionCalculator
	{

		private static final long serialVersionUID = 1L;

		/**
		 * Activation of the output layer from the feedforward phase
		 */
		@Constant
		protected float[] ffActivation;
		protected final int activationStartPosition;
		protected final int activationRowStep;
		protected final int activationColumnStep;

		/**
		 * Weight updates array
		 */
		protected final float[] weightUpdates;

		protected float learningRate;
		protected final float momentum;
		protected final float l1weightDecay;
		protected final float l2weightDecay;

		private final int winnersStartPosition;
		private final int[] maxoutWinners;

		@SuppressWarnings("unused")
		public AparapiBackpropMaxout(Connections inputConnection, ValuesProvider valuesProvider, ValuesProvider activations, List<Tensor> weightUpdates, float learningRate, float momentum,
				float l1weightDecay, float l2weightDecay)
		{
			super(inputConnection, valuesProvider, inputConnection.getOutputLayer());

			Matrix m = TensorFactory.tensor(inputConnection.getInputLayer(), inputConnection, activations);
			this.ffActivation = m.getElements();
			this.activationStartPosition = m.getStartIndex();
			this.activationRowStep = m.getRowElementsDistance();
			this.activationColumnStep = m.getColumnElementsDistance();

			this.learningRate = momentum;
			this.momentum = momentum;
			this.l1weightDecay = l1weightDecay;
			this.l2weightDecay = l2weightDecay;
			this.weightUpdates = weightUpdates.get(0).getElements();

			this.winnersStartPosition = MaxoutWinners.getInstance().getStartPositions(inputConnection);
			this.maxoutWinners = MaxoutWinners.getInstance().getWinners();
		}

		@Override
		public void run()
		{
			int id = getGlobalId();

			int maxoutId = 0, weightId = 0;
			float weight = 0, weightUpdate = 0;

			// each input example
			for (int i = 0; i < miniBatchSize; i++)
			{
				// each connection (of the combined connections)
				maxoutId = maxoutWinners[winnersStartPosition + id * miniBatchSize + i];
				weightId = weightStartPosition + weightsInitialStep * id + maxoutId * weightsStep;
				weight = weights[weightId];

				weightUpdate += output[outputStartPosition + id * outputRowStep + i * outputColumnStep] * ffActivation[activationStartPosition + maxoutId * activationRowStep + i * activationColumnStep];
				weightUpdate = learningRate * weightUpdate + momentum * weightUpdates[weightId] - l1weightDecay * abs(weight) - l2weightDecay * weight * weight;
				weights[weightId] += weightUpdate;
				weightUpdates[weightId] = weightUpdate;

				input[activationStartPosition + maxoutId * activationRowStep + i * activationColumnStep] += output[outputStartPosition + id * outputRowStep + i * outputColumnStep];
			}
		}

		@Override
		public ValuesProvider getActivations()
		{
			return null;
		}

		@Override
		public void setActivations(ValuesProvider activations)
		{
		}
	}
}
