package com.github.neuralnetworks.training.backpropagation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorTensorFunctions;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.TensorFunction.TensorFunctionDerivative;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Connection calculator for the backpropagation phase of the algorithm
 * The difference with the regular ConnectionCalculatorImpl is that forwardBackprop's and backwardBackprop's properties (learing rate, momentum, weight decay) are updated before each propagation
 */
public abstract class BackPropagationConnectionCalculatorImpl implements BackPropagationConnectionCalculator, ConnectionCalculatorTensorFunctions
{
	private static final long serialVersionUID = -8854054073444883314L;

	private Properties properties;
	protected transient Map<Connections, BackPropagationConnectionCalculator> connectionCalculators;
	protected transient List<ConnectionCalculator> inputFunctions;
	protected transient ValuesProvider activations;
	protected Layer currentLayer;
	protected int miniBatchSize;

	protected List<TensorFunction> inputModifierFunctions;
	protected List<TensorFunction> activationFunctions;

	public BackPropagationConnectionCalculatorImpl(Properties properties)
	{
		init();
		this.properties = properties;
		this.inputModifierFunctions = new ArrayList<>();
		this.activationFunctions = new ArrayList<>();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	protected void init()
	{
		this.inputFunctions = new ArrayList<>();
		this.connectionCalculators = new HashMap<>();
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		List<Connections> chunk = connections.stream().filter(c -> !connectionCalculators.containsKey(c) || targetLayer != currentLayer || miniBatchSize != TensorFactory.batchSize(valuesProvider))
				.collect(Collectors.toList());

		if (chunk.size() > 0)
		{
			miniBatchSize = TensorFactory.batchSize(valuesProvider);
			currentLayer = targetLayer;
			addBackpropFunction(chunk, connectionCalculators, valuesProvider, activations, targetLayer);
			inputFunctions.clear();
			inputFunctions.addAll(connectionCalculators.values());
		}

		List<Connections> chunkCalc = new ArrayList<>();
		for (ConnectionCalculator cc : inputFunctions)
		{
			BackPropagationConnectionCalculator bc = (BackPropagationConnectionCalculator) cc;

			chunkCalc.clear();

			Layer target = targetLayer;
			for (Connections c : connections)
			{
				if (connectionCalculators.get(c) == bc)
				{
					chunkCalc.add(c);
					if (Util.isBias(c.getInputLayer()) && c.getInputLayer() != targetLayer && !(OperationsFactory.isBPMaxout(bc)))
					{
						target = c.getInputLayer();
					}
				}
			}

			if (chunkCalc.size() > 0)
			{
				if (getInputModifierFunctions() != null) {
					getInputModifierFunctions().stream().filter(f -> f instanceof TensorFunctionDerivative).forEach(f -> ((TensorFunctionDerivative) f).setActivations(TensorFactory.tensor(targetLayer, chunkCalc, getActivations())));
				}

				calculateInputModifierFunctions(chunkCalc, targetLayer, valuesProvider);

				bc.setActivations(getActivations());
				bc.calculate(chunkCalc, valuesProvider, target);

				if (getActivationFunctions() != null) {
					getActivationFunctions().stream().filter(f -> f instanceof TensorFunctionDerivative).forEach(f -> ((TensorFunctionDerivative) f).setActivations(TensorFactory.tensor(targetLayer, chunkCalc, getActivations())));
				}

				calculateActivationFunctions(chunkCalc, targetLayer, valuesProvider);
			}
		}
	}

	protected abstract void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer);

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public NeuralNetwork getNeuralNetwork()
	{
		return properties.getParameter(Constants.NEURAL_NETWORK);
	}

	@Override
	public ValuesProvider getActivations()
	{
		return activations;
	}

	@Override
	public void setActivations(ValuesProvider activations)
	{
		this.activations = activations;
	}

	protected ValuesProvider getWeightUpdates()
	{
		return properties.getParameter(Constants.WEIGHT_UDPATES);
	}

	@Override
	public List<TensorFunction> getInputModifierFunctions()
	{
		return inputModifierFunctions;
	}

	@Override
	public void setInputModifierFunctions(List<TensorFunction> inputModifierFunctions)
	{
		this.inputModifierFunctions = inputModifierFunctions; 
	}

	@Override
	public List<TensorFunction> getActivationFunctions()
	{
		return activationFunctions;
	}

	@Override
	public void setActivationFunctions(List<TensorFunction> activationFunctions)
	{
		this.activationFunctions = activationFunctions;
	}

	@Override
	public List<ConnectionCalculator> getInputFunctions()
	{
		return inputFunctions;
	}

	@Override
	public void setInputFunctions(List<ConnectionCalculator> inputFunctions)
	{
		this.inputFunctions = inputFunctions;
	}
}
