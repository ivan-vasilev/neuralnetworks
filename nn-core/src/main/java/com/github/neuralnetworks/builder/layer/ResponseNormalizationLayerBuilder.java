package com.github.neuralnetworks.builder.layer;

import java.util.Arrays;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * @author tmey
 */
public class ResponseNormalizationLayerBuilder extends NamedSingleInputLayerBuilder
{
	private float k = 2;
	private int n = 5;
	private float a = 0.01f;
	private float b = 1f;

	public ResponseNormalizationLayerBuilder(String default_layer_name)
	{
		super(default_layer_name);
	}

	@Override
	protected Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{
		Layer newLayer = new Layer();

		// create connection and calculator
		new RepeaterConnection(inputLayer, newLayer, inputLayer.getNeuronCount());

		ConnectionCalculator cc = OperationsFactory.lrnConnectionCalculator(k, n, a, b);

		((LayerCalculatorImpl) neuralNetwork.getLayerCalculator()).addConnectionCalculator(newLayer, cc);

		// set basic information for the next layer
		newLayer.setLayerDimension(Arrays.copyOf(inputLayer.getLayerDimension(), inputLayer.getLayerDimension().length));
		newLayer.setName(newLayerName);

		// add new layer to the network
		neuralNetwork.addLayer(newLayer);

		return newLayer;
	}

	public void setK(float k)
	{
		if (k < 0)
		{
			throw new IllegalArgumentException("k must be equals or greater than zero!");
		}

		this.k = k;
	}

	public void setN(int n)
	{
		if (n < 0)
		{
			throw new IllegalArgumentException("n must be equals or greater than zero!");
		}

		this.n = n;
	}

	public void setA(float a)
	{
		if (a < 0)
		{
			throw new IllegalArgumentException("a must be equals or greater than zero!");
		}

		this.a = a;
	}

	public void setB(float b)
	{
		if (b < 0)
		{
			throw new IllegalArgumentException("b must be equals or greater than zero!");
		}

		this.b = b;
	}
}
