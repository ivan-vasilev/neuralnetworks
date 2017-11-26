package com.github.neuralnetworks.training.backpropagation;

import java.util.List;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorBase;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.operations.TensorFunction.TensorFunctionDerivative;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Util;

/**
 * Implementation of the backpropagation algorithm
 */
public class BackPropagationLayerCalculatorImpl extends LayerCalculatorBase implements BackPropagationLayerCalculator
{

	private static final long serialVersionUID = 1L;

	private ValuesProvider activations;
	private TensorFunctionDerivative outputDerivative;
	private boolean skipEndLayers;

	public BackPropagationLayerCalculatorImpl()
	{
		super();
		this.skipEndLayers = true;
	}

	@Override
	public void backpropagate(NeuralNetwork nn, ValuesProvider activations, ValuesProvider results)
	{
		this.activations = activations;

		List<ConnectionCandidate> connections = new BreadthFirstOrderStrategy(nn, nn.getOutputLayer()).order();

		connections.forEach(c -> c.target = Util.getOppositeLayer(c.connection, c.target));

		if (skipEndLayers)
		{
			connections.removeIf(c -> Util.isBias(c.target) || nn.getInputLayer() == c.target);
		}

		if (outputDerivative != null)
		{
			outputDerivative.setActivations(TensorFactory.tensor(nn.getOutputLayer(), nn.getOutputLayer().getConnections(), activations));
			outputDerivative.value(TensorFactory.tensor(nn.getOutputLayer(), nn.getOutputLayer().getConnections(), results));
		}

		calculate(results, connections, nn);
	}

	@Override
	public ConnectionCalculator getConnectionCalculator(Layer layer)
	{
		ConnectionCalculator cc = super.getConnectionCalculator(layer);
		if (cc instanceof BackPropagationConnectionCalculator)
		{
			((BackPropagationConnectionCalculator) cc).setActivations(activations);
		}

		return cc;
	}

	public TensorFunctionDerivative getOutputDerivative()
	{
		return outputDerivative;
	}

	public void setOutputDerivative(TensorFunctionDerivative outputDerivative)
	{
		this.outputDerivative = outputDerivative;
	}

	public boolean getSkipEndLayers()
	{
		return skipEndLayers;
	}

	public void setSkipEndLayers(boolean skipEndLayers)
	{
		this.skipEndLayers = skipEndLayers;
	}
}