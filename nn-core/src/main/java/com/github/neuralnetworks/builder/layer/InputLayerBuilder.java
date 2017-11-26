package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.LayerUtil;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.structure.NamedLayerBuilder;
import com.github.neuralnetworks.training.Hyperparameters;

/**
 * The input layer must have layer calculator because this one is used during back propagation for the bias layer that is directly connected to the output layer!
 * (See TrainerFactory: line 119 (public static BackPropagationLayerCalculatorImpl bplc(NeuralNetworkImpl nn, Properties p)))
 *
 * @author tmey
 */
public class InputLayerBuilder extends NamedLayerBuilder
{

	private int width;
	private int height;
	private int featureMaps;

	public InputLayerBuilder(String name, int width, int height, int featureMaps)
	{
		super("InputLayer");
		if (width <= 0)
		{
			throw new IllegalArgumentException("width must be greater than 0! (" + width + ")");
		}

		if (height <= 0)
		{
			throw new IllegalArgumentException("height must be greater than 0! (" + width + ")");
		}

		if (featureMaps <= 0)
		{
			throw new IllegalArgumentException("featureMaps must be greater than 0! (" + width + ")");
		}

		this.setName(name);
		this.width = width;
		this.height = height;
		this.featureMaps = featureMaps;
	}


	@Override
	public Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Hyperparameters hyperparameters)
	{

		// initialize and add layer

		Layer inputLayer = new Layer(newLayerName);
		inputLayer.setLayerDimension(new int[] { height, width, featureMaps });

		neuralNetwork.addLayer(inputLayer);

		if (neuralNetwork.getLayerCalculator() != null)
		{
			// change transfer and activation function
			LayerUtil.changeActivationAndTransferFunction(neuralNetwork.getLayerCalculator(), inputLayer, TransferFunctionType.WeightedSum, ActivationType.Nothing);
		}

		return inputLayer;
	}

	@Override
	public String toString()
	{
		return "InputLayerBuilder{" +
				"name=" + (getName() != null ? getName() : DEFAULT_LAYER_NAME) +
				", width=" + width +
				", height=" + height +
				", featureMaps=" + featureMaps +
				'}';
	}
}
