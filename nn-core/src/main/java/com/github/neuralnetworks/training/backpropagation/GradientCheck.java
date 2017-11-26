package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.util.Constants;

/**
 * Check bp gradients
 */
public class GradientCheck implements Serializable
{
	private static final long serialVersionUID = 1L;

	private ValuesProvider deltas;
	private float e;
	private BackPropagationTrainer<?> trainer;

	public GradientCheck(BackPropagationTrainer<?> trainer, float e)
	{
		super();
		this.trainer = trainer;
		this.e = e;
	}

	public void compute()
	{
		ValuesProvider gradients = trainer.getProperties().getParameter(Constants.WEIGHT_UDPATES);
		deltas = TensorFactory.duplicate(gradients);

		NeuralNetworkImpl nn = (NeuralNetworkImpl) trainer.getNeuralNetwork();
		ValuesProvider weights = nn.getProperties().getParameter(Constants.WEIGHTS_PROVIDER);
		ValuesProvider originalWeights = TensorFactory.duplicate(weights);
		TensorFactory.copy(weights, originalWeights);

		Hyperparameters originalHP = trainer.getHyperparameters();

		// fake hyperparameters
		Hyperparameters hp = new Hyperparameters();
		trainer.setHyperparameters(hp);
		hp.setDefaultLearningRate(trainer.getTrainingBatchSize());

		trainer.train();

		TensorFactory.copy(originalWeights, weights);
		trainer.setSkipBackprop(true);

		for (Connections c : trainer.getNeuralNetwork().getConnections())
		{
			if (c instanceof WeightsConnections)
			{
				Tensor delta = deltas.get(c);
				TensorIterator deltaIt = delta.iterator();

				Tensor w = ((WeightsConnections) c).getWeights();
				TensorIterator it = w.iterator();

				Tensor gradient = gradients.get(c);

				while (it.hasNext())
				{
					int i = it.next();
					int[] pos = it.getCurrentPosition();

					float weight = w.getElements()[i];

					// +e phase
					w.getElements()[i] = weight + e;
					trainer.train();
					float lossA = trainer.getLossFunctionCurrentValue();

					// -e phase
					w.getElements()[i] = weight - e;
					trainer.train();
					float lossB = trainer.getLossFunctionCurrentValue();

					// aggregate values
					float v = (lossA - lossB) / (2 * e);

					delta.getElements()[deltaIt.next()] = Math.abs(v - gradient.get(pos));

					// restore weights
					w.getElements()[i] = weight;
				}
			}
		}

		trainer.setHyperparameters(originalHP);
	}

	public ValuesProvider getDeltas()
	{
		return deltas;
	}

	public float getMaxDelta()
	{
		return getMax(deltas);
	}
	
	public float getMaxGradient()
	{
		return getMax(trainer.getProperties().getParameter(Constants.WEIGHT_UDPATES));
	}
	
	private float getMax(ValuesProvider provider)
	{
		float max = Float.MIN_VALUE;
		NeuralNetworkImpl nn = (NeuralNetworkImpl) trainer.getNeuralNetwork();
		for (Connections c : nn.getConnections())
		{
			if (c instanceof WeightsConnections)
			{
				Tensor d = provider.get(c);
				max = Math.max(max, d.getElements()[TensorFactory.max(d)]);
			}
		}
		
		return max;
	}

	public float getMinDelta()
	{
		return getMin(deltas);
	}
	
	public float getMinGradient()
	{
		return getMin(trainer.getProperties().getParameter(Constants.WEIGHT_UDPATES));
	}
	
	public float getMin(ValuesProvider provider)
	{
		float min = Float.MAX_VALUE;
		NeuralNetworkImpl nn = (NeuralNetworkImpl) trainer.getNeuralNetwork();
		for (Connections c : nn.getConnections())
		{
			if (c instanceof WeightsConnections)
			{
				Tensor d = provider.get(c);
				min = Math.min(min, d.getElements()[TensorFactory.min(d)]);
			}
		}
		
		return min;
	}

	public float getAverageDelta()
	{
		float average = 0;
		int count = 0;
		NeuralNetworkImpl nn = (NeuralNetworkImpl) trainer.getNeuralNetwork();
		for (Connections c : nn.getConnections())
		{
			if (c instanceof WeightsConnections)
			{
				Tensor d = deltas.get(c);
				TensorIterator it = d.iterator();
				while (it.hasNext())
				{
					int i = it.next();
					average += Math.abs(d.getElements()[i]);
					count++;
				}
			}
		}

		return average / count;
	}
}
