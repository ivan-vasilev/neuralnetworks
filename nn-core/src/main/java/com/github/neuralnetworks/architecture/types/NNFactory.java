package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Factory class for neural networks
 */
public class NNFactory
{

	/**
	 * Create convolutional network
	 * 
	 * @param layers
	 *          The first layer must have 3 parameters - rows, columns and filter count (usually 1)
	 *          Convolutional connections must have 7 parameters - kernelRows, kernelColumns, filters, rowStride, columnStride, rowPadding, and columnPadding. The first layer must be convolutional.
	 *          Subsampling connections must have 6 parameters - subsamplingRegionRows, subsamplingRegionCols, rowStride, columnStride, rowPadding, and columnPadding.
	 *          Regular layers must have 1 parameter - neuron count
	 * 
	 * @param addBias
	 * @return neural network
	 */
	public static NeuralNetworkImpl convNN(int[][] layers, boolean addBias)
	{
		if (layers.length <= 1)
		{
			throw new IllegalArgumentException("more than one layer is required");
		}

		if (layers[0].length != 3)
		{
			throw new IllegalArgumentException("first layer must be convolutional");
		}

		NeuralNetworkImpl result = new NeuralNetworkImpl();
		ConnectionFactory cf = new ConnectionFactory();
		result.setProperties(new Properties());
		result.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

		Layer prev = null;
		int prevUnitCount = layers[0][0] * layers[0][1] * layers[0][2];
		result.addLayer(prev = new Layer());
		prev.setName("INPUT");

		for (int i = 1; i < layers.length; i++)
		{
			int[] l = layers[i];
			Layer newLayer = null;
			Layer biasLayer = null;
			if (l.length == 1)
			{
				cf.fullyConnected(prev, newLayer = new Layer(), prevUnitCount, l[0]);
				newLayer.setName("FC_" + i);
				if (addBias)
				{
					cf.fullyConnected(biasLayer = new Layer(), newLayer, 1, l[0]);
					biasLayer.setName(newLayer.getName() + "_B");
				}

				prevUnitCount = l[0];
			} else if (l.length == 7 || l.length == 6)
			{
				Integer inputFMRows = null;
				Integer inputFMCols = null;
				Integer filters = null;
				if (i == 1)
				{
					inputFMRows = layers[0][0];
					inputFMCols = layers[0][1];
					filters = layers[0][2];
				} else
				{
					for (Connections c : prev.getConnections())
					{
						if (c.getOutputLayer() == prev)
						{
							if (c instanceof Conv2DConnection)
							{
								Conv2DConnection cc = (Conv2DConnection) c;
								inputFMRows = cc.getOutputFeatureMapRowsWithPadding();
								inputFMCols = cc.getOutputFeatureMapColumnsWithPadding();
								filters = cc.getOutputFilters();
								break;
							} else if (c instanceof Subsampling2DConnection)
							{
								Subsampling2DConnection sc = (Subsampling2DConnection) c;
								inputFMRows = sc.getOutputFeatureMapRowsWithPadding();
								inputFMCols = sc.getOutputFeatureMapColumnsWithPadding();
								filters = sc.getFilters();
								break;
							}
						}
					}
				}

				if (l.length == 7)
				{
					Conv2DConnection c = cf.conv2d(prev, newLayer = new Layer(), inputFMRows, inputFMCols, filters, l[0], l[1], l[2], l[3], l[4], l[5], l[6]);
					newLayer.setName("CONV_" + i);

					if (addBias)
					{
						cf.conv2d(biasLayer = new Layer(), newLayer, c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), 1, 1, 1, l[2], 1, 1, l[5], l[6]);
						biasLayer.setName(newLayer.getName() + "_B");
					}

					prevUnitCount = c.getOutputUnitCountWithPadding();
				} else if (l.length == 6)
				{
					Subsampling2DConnection c = cf.subsampling2D(prev, newLayer = new Layer(), inputFMRows, inputFMCols, l[0], l[1], filters, l[2], l[3], l[4], l[5]);
					newLayer.setName("SUB_" + i);

					prevUnitCount = c.getOutputUnitCountWithPadding();
				}
			}

			result.addLayer(newLayer);
			if (biasLayer != null)
			{
				result.addLayer(biasLayer);
			}

			prev = newLayer;
		}

		return result;
	}

	/**
	 * Multi layer perceptron with fully connected layers
	 * 
	 * @param layers
	 *          - neuron count for each layer
	 * @param addBias
	 * @return
	 */
	public static NeuralNetworkImpl mlp(int[] layers, boolean addBias)
	{
		NeuralNetworkImpl result = new NeuralNetworkImpl();
		mlp(result, new ConnectionFactory(), layers, addBias);
		return result;
	}

	public static void mlp(NeuralNetworkImpl nn, ConnectionFactory cf, int[] layers, boolean addBias)
	{
		if (layers.length <= 1)
		{
			throw new IllegalArgumentException("more than one layer is required");
		}

		if (nn.getProperties() == null)
		{
			nn.setProperties(new Properties());
		}

		nn.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

		addFullyConnectedLayer(nn, new Layer("0"), cf, layers[0], layers[0], addBias);
		for (int i = 1; i < layers.length; i++)
		{
			addFullyConnectedLayer(nn, new Layer("" + i), cf, layers[i - 1], layers[i], addBias);
		}
	}

	/**
	 * Add fully connected layer to the output layer of the network
	 * 
	 * @param nn
	 * @param layer
	 * @param addBias
	 */
	public static FullyConnected addFullyConnectedLayer(NeuralNetworkImpl nn, Layer layer, ConnectionFactory cf, int inputUnitCount, int outputUnitCount, boolean addBias)
	{
		FullyConnected result = null;
		if (nn.addLayer(layer) && nn.getOutputLayer() != layer)
		{
			result = cf.fullyConnected(nn.getOutputLayer(), layer, inputUnitCount, outputUnitCount);
		}

		if (addBias && nn.getInputLayer() != layer)
		{
			nn.addConnections(cf.fullyConnected(new Layer("bias layer to " + layer.getName()), layer, 1, outputUnitCount));
		}

		return result;
	}

	public static RBM rbm(int visibleCount, int hiddenCount, boolean addBias)
	{
		RBM result = new RBM();
		ConnectionFactory cf = new ConnectionFactory();
		result.addConnections(cf.fullyConnected(new Layer(), new Layer(), visibleCount, hiddenCount));

		if (addBias)
		{
			result.addConnections(cf.fullyConnected(new Layer(), result.getVisibleLayer(), 1, visibleCount));
			result.addConnections(cf.fullyConnected(new Layer(), result.getHiddenLayer(), 1, hiddenCount));
		}

		return result;
	}

	public static DBN dbn(int[] layers, boolean addBias)
	{
		if (layers.length <= 1)
		{
			throw new IllegalArgumentException("more than one layer is required");
		}

		DBN result = new DBN();
		ConnectionFactory cf = new ConnectionFactory();
		result.setProperties(new Properties());
		result.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

		result.addLayer(new Layer());
		for (int i = 1; i < layers.length; i++)
		{
			RBM rbm = new RBM();
			rbm.setProperties(new Properties());
			rbm.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

			rbm.addConnections(cf.fullyConnected(result.getOutputLayer(), new Layer(), layers[i - 1], layers[i]));

			if (addBias)
			{
				rbm.addConnections(cf.fullyConnected(new Layer(), rbm.getVisibleLayer(), 1, layers[i - 1]));
				rbm.addConnections(cf.fullyConnected(new Layer(), rbm.getHiddenLayer(), 1, layers[i]));
			}

			result.addNeuralNetwork(rbm);
		}

		return result;
	}

	public static StackedAutoencoder sae(int[] layers, boolean addBias)
	{
		if (layers == null || layers.length <= 1)
		{
			throw new IllegalArgumentException("more than one layer is required");
		}

		ConnectionFactory cf = new ConnectionFactory();
		Properties properties = new Properties();
		properties.setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());
		StackedAutoencoder result = new StackedAutoencoder(new Layer());
		result.setProperties(properties);

		for (int i = 1; i < layers.length; i++)
		{
			Autoencoder ae = new Autoencoder();
			ae.setProperties(new Properties());
			ae.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

			ae.addLayer(result.getOutputLayer());
			NNFactory.addFullyConnectedLayer(ae, new Layer(), cf, layers[i - 1], layers[i], addBias);
			NNFactory.addFullyConnectedLayer(ae, new Layer(), cf, layers[i], layers[i - 1], addBias);

			result.addNeuralNetwork(ae);
		}

		return result;
	}

	public static void populateBiasLayers(LayerCalculatorImpl lc, NeuralNetwork nn)
	{
		nn.getLayers().stream().filter(l -> Util.isBias(l)).forEach(l -> lc.addConnectionCalculator(l, new ConstantConnectionCalculator()));
	}
}
