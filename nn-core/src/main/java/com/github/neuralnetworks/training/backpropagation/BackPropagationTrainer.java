package com.github.neuralnetworks.training.backpropagation;


import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Base backpropagation one step trainer
 * It has two additional parameters:
 * BackPropagationLayerCalculator for the backpropagation phase
 * OutputErrorDerivative for calculating the derivative of the output error
 * This allows for various implementations of these calculators to be used (for example via GPU or other)
 */
public class BackPropagationTrainer<N extends NeuralNetwork> extends OneStepTrainer<N>
{

	private static final long serialVersionUID = 1L;

	protected ValuesProvider activations;
	protected ValuesProvider backpropagation;
	protected TrainingInputData input;
	protected boolean skipBackprop;
	protected Map<Connections, WeightUpdates> weightUpdates;

	public BackPropagationTrainer(Properties properties)
	{
		super(properties);

		NeuralNetwork nn = getNeuralNetwork();

		activations = TensorFactory.tensorProvider(nn, getTrainingBatchSize(), Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		activations.add(getLossFunction(), activations.get(getNeuralNetwork().getOutputLayer()).getDimensions());
		backpropagation = TensorFactory.tensorProvider(nn, getTrainingBatchSize(), Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		weightUpdates = new HashMap<>();
	}

	@Override
	public void reset()
	{
		skipBackprop = false;

		if (getTrainingInputProvider() != null)
		{
			getTrainingInputProvider().reset();
		}

		activations.getTensors().forEach(t -> t.forEach(i -> t.getElements()[i] = 0));
		backpropagation.getTensors().forEach(t -> t.forEach(i -> t.getElements()[i] = 0));
		ValuesProvider weightUpdates = (ValuesProvider) properties.get(Constants.WEIGHT_UDPATES);
		weightUpdates.getTensors().forEach(t -> t.forEach(i -> t.getElements()[i] = 0));
	}

	@Override
	public void train()
	{
		super.train();

		removeDropout();
	}
	/*
	 * (non-Javadoc)
	 * 
	 * @see com.github.neuralnetworks.training.OneStepTrainer#learnInput(com.github.neuralnetworks.training.TrainingInputData)
	 * The training example is propagated forward through the network (via the LayerCalculator lc) and the results are stored.
	 * After that the error is backpropagated (via BackPropagationLayerCalculator blc).
	 */
	@Override
	protected void learnInput(int batch)
	{
		// forward
		NeuralNetwork nn = getNeuralNetwork();
		Set<Layer> calculatedLayers = new UniqueList<Layer>();
		calculatedLayers.add(nn.getInputLayer());
		nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, activations);

		// backward
		if (!skipBackprop)
		{
			LossFunction d = getLossFunction();
			d.getLossFunctionDerivative(activations.get(nn.getOutputLayer()), activations.get(d), backpropagation.get(nn.getOutputLayer()));

			triggerEvent(new LossFunctionEvent(this, activations.get(nn.getOutputLayer()), activations.get(d), backpropagation.get(nn.getOutputLayer())));

			BackPropagationLayerCalculator blc = getBPLayerCalculator();
			blc.backpropagate(nn, activations, backpropagation);

			updateWeights(activations, backpropagation, (ValuesProvider) properties.get(Constants.WEIGHT_UDPATES));
		}
	}

	@Override
	protected TrainingInputData getInput()
	{
		if (input == null)
		{
			input = new TrainingInputDataImpl(activations.get(getNeuralNetwork().getInputLayer()), activations.get(getProperties().getParameter(Constants.LOSS_FUNCTION)));
		}

		return input;
	}

	protected void updateWeights(ValuesProvider activations, ValuesProvider backpropagation, ValuesProvider weightUpdatesVP)
	{
		getNeuralNetwork().getConnections().stream().filter(c -> c instanceof WeightsConnections).forEach(c -> {
			WeightUpdates wu = weightUpdates.get(c);
			if (wu == null)
			{
				weightUpdates.put(c, wu = OperationsFactory.weightUpdates((WeightsConnections) c, backpropagation, activations, weightUpdatesVP.get(c)));
			}

			Hyperparameters hp = getHyperparameters();
			wu.updateWeights(hp.getLearningRate(c), hp.getMomentum(c), hp.getL1WeightDecay(c), hp.getL2WeightDecay(c));
		});
	}

	public void removeDropout()
	{
		boolean hasDropout = false;
		NeuralNetwork nn = getNeuralNetwork();
		for (Connections cs : nn.getConnections().stream().filter(c -> c instanceof FullyConnected && c.getOutputLayer() != nn.getOutputLayer() && !Util.isBias(c.getInputLayer()))
				.collect(Collectors.toList()))
		{
			if (getHyperparameters().getDropoutRate(cs) > 0)
			{
				hasDropout = true;
				break;
			}
		}

		if (hasDropout)
		{
			LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
			nn.getConnections().stream().filter(c -> c instanceof FullyConnected && c.getInputLayer() != nn.getInputLayer() && !Util.isBias(c.getInputLayer())).forEach(c -> {
				if (lc.getConnectionCalculator(c.getInputLayer()) instanceof ConnectionCalculatorImpl)
				{
					ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getConnectionCalculator(c.getInputLayer());
					if (cc.getActivationFunctions().stream().filter(f -> OperationsFactory.isNoiseMask(f)).findAny().isPresent())
					{
						Hyperparameters hp = getHyperparameters();
						cc.getActivationFunctions().removeIf(f -> OperationsFactory.isNoiseMask(f));
						FullyConnected fc = (FullyConnected) c;
						fc.getWeights().forEach(i -> fc.getWeights().getElements()[i] = fc.getWeights().getElements()[i] * (1 - hp.getDropoutRate(fc)));
					}
				}
			});
		}
	}

	public BackPropagationLayerCalculator getBPLayerCalculator()
	{
		return getProperties().getParameter(Constants.BACKPROPAGATION);
	}

	public void setBPLayerCalculator(BackPropagationLayerCalculator bplc)
	{
		getProperties().setParameter(Constants.BACKPROPAGATION, bplc);
	}

	public LossFunction getLossFunction()
	{
		return getProperties().getParameter(Constants.LOSS_FUNCTION);
	}

	public void setLossFunction(LossFunction lossFunction)
	{
		getProperties().setParameter(Constants.LOSS_FUNCTION, lossFunction);
	}

	@Override
	public ValuesProvider getActivations()
	{
		return activations;
	}

	public ValuesProvider getBackpropagation()
	{
		return backpropagation;
	}

	public void setSkipBackprop(boolean skipBackprop)
	{
		this.skipBackprop = skipBackprop;
	}
	
	public Tensor getCurrentNetworkOutput()
	{
		return activations.get(getNeuralNetwork().getOutputLayer());
	}

	public Tensor getLossFunctionCurrentDerivative()
	{
		return backpropagation.get(getNeuralNetwork().getOutputLayer());
	}

	public Map<Connections, WeightUpdates> getWeightUpdates()
	{
		return weightUpdates;
	}

	public float getLossFunctionCurrentValue()
	{
		LossFunction d = getLossFunction();
		return d.getLossFunction(activations.get(getNeuralNetwork().getOutputLayer()), activations.get(d));
	}

	public Tensor getCurrentActivations()
	{
		return activations.get(getNeuralNetwork().getOutputLayer());
	}

	public static class LossFunctionEvent extends TrainingEvent
	{

		private static final long serialVersionUID = 1L;

		private Tensor activation;
		private Tensor target;
		private Tensor result;

		public LossFunctionEvent(Trainer<?> source, Tensor activation, Tensor target, Tensor result)
		{
			super(source);
			this.activation = activation;
			this.target = target;
			this.result = result;
		}

		public Tensor getActivation()
		{
			return activation;
		}

		public Tensor getTarget()
		{
			return target;
		}

		public Tensor getResult()
		{
			return result;
		}
	}
}
