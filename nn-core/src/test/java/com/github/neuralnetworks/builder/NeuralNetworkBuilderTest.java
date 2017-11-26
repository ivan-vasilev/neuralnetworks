package com.github.neuralnetworks.builder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.NormalDistributionInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.Properties;

public class NeuralNetworkBuilderTest
{

	public void testBuild() throws Exception
	{

		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();

		neuralNetworkBuilder.addLayerBuilder(new InputLayerBuilder("input", 250, 250, 3));

		ConvolutionalLayerBuilder convLayerBuilder = new ConvolutionalLayerBuilder(3, 5);
		convLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(convLayerBuilder);

		PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
		poolingLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(poolingLayerBuilder);

		neuralNetworkBuilder.addLayerBuilder(new FullyConnectedLayerBuilder(10));

		neuralNetworkBuilder.build();

	}

	public void testLayerBuilder()
	{
		NeuralNetworkImpl neuralNetwork = new NeuralNetworkImpl();
		ConnectionFactory cf = new ConnectionFactory();
		neuralNetwork.setProperties(new Properties());
		neuralNetwork.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());
		neuralNetwork.getProperties().setParameter(Constants.CONNECTION_FACTORY, cf);

		Layer inputLayer = new InputLayerBuilder("input", 250, 250, 3).build(neuralNetwork);

		System.out.println(Arrays.toString(inputLayer.getLayerDimension()));
		assertEquals(250 * 250 * 3, inputLayer.getNeuronCount());

		ConvolutionalLayerBuilder convLayerBuilder = new ConvolutionalLayerBuilder(3, 5);
		convLayerBuilder.setPaddingSize(2);
		convLayerBuilder.setStrideSize(3);
		convLayerBuilder.setAddBias(false);
		Layer convLayer = convLayerBuilder.build(neuralNetwork);

		System.out.println(Arrays.toString(convLayer.getLayerDimension()));
		assertEquals((83 + 2 + 2) * (83 + 2 + 2) * 5, convLayer.getNeuronCount());

		PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
		poolingLayerBuilder.setPaddingSize(2);
		Layer poolingLayer = poolingLayerBuilder.build(neuralNetwork);

		System.out.println(Arrays.toString(poolingLayer.getLayerDimension()));
		assertEquals((43 + 2 + 2) * (43 + 2 + 2) * 5, poolingLayer.getNeuronCount());

		Layer fullyLayer = new FullyConnectedLayerBuilder(10).build(neuralNetwork);

		System.out.println(Arrays.toString(fullyLayer.getLayerDimension()));
		assertEquals(10, fullyLayer.getNeuronCount());
	}

	public void testTrainerAndLayerBuilder()
	{
		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();

		// set parameter
		neuralNetworkBuilder.setDropoutRate(0.1f);
		neuralNetworkBuilder.setLearningRate(0.5f);
		neuralNetworkBuilder.setMomentum(0.5f);


		// add layer builder

		neuralNetworkBuilder.addLayerBuilder(new InputLayerBuilder("input", 250, 250, 3));

		ConvolutionalLayerBuilder convLayerBuilder = new ConvolutionalLayerBuilder(3, 5);
		convLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(convLayerBuilder);

		PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
		poolingLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(poolingLayerBuilder);

		neuralNetworkBuilder.addLayerBuilder(new FullyConnectedLayerBuilder(10));
//        neuralNetworkTrainerPair.getRight().train();
	}

	public void testRandomSeedResetFunction()
	{

		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();

		neuralNetworkBuilder.addLayerBuilder(new InputLayerBuilder("input", 250, 250, 3));

		ConvolutionalLayerBuilder convLayerBuilder = new ConvolutionalLayerBuilder(3, 5);
		convLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(convLayerBuilder);
		convLayerBuilder.setWeightInitializer(new NormalDistributionInitializer());

		PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
		poolingLayerBuilder.setPaddingSize(2);
		neuralNetworkBuilder.addLayerBuilder(poolingLayerBuilder);

		neuralNetworkBuilder.addLayerBuilder(new FullyConnectedLayerBuilder(10));
		neuralNetworkBuilder.setRand(new NNRandomInitializer(new RandomInitializerImpl(0, 1)));


		// create network
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> trainerOne = neuralNetworkBuilder.buildWithTrainer();

		// reseet
		assertTrue(neuralNetworkBuilder.resetRandomSeed());

		// create second network
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> trainerTwo = neuralNetworkBuilder.buildWithTrainer();

		// compare weights

		ValuesProvider conf1Weights = ((NeuralNetworkImpl) ((BackPropagationTrainer<?>) trainerOne.getRight()).getNeuralNetwork()).getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		ValuesProvider conf2Weights = ((NeuralNetworkImpl) ((BackPropagationTrainer<?>) trainerTwo.getRight()).getNeuralNetwork()).getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		Iterator<Tensor> itw1 = conf1Weights.getTensors().iterator();
		Iterator<Tensor> itw2 = conf2Weights.getTensors().iterator();

		while (itw1.hasNext() && itw2.hasNext())
		{


			Tensor t1 = itw1.next();
			Tensor t2 = itw2.next();

			Tensor.TensorIterator t1it = t1.iterator();
			Tensor.TensorIterator t2it = t2.iterator();
			while (t1it.hasNext() && t2it.hasNext())
			{
				int t1index = t1it.next();
				int t2index = t2it.next();
				if (t1.getElements() == t2.getElements())
				{
					throw new IllegalStateException("the activation arrays for backpropagation must be different instances!");
				}
				if (t1.getElements()[t1index] == Float.NaN)
				{
					throw new IllegalStateException("The first network contains a activation for backpropagation with the value NaN!");
				}
				if (t2.getElements()[t2index] == Float.NaN)
				{
					throw new IllegalStateException("The second network contains a activation for backpropagation with the value NaN!");
				}

				assertEquals("One weight is different than ", 0, Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]), 0);

			}

		}
	}


	@Test
	public void testConstruction()
	{

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		Random r = new Random(123456789);
		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));

			// first part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 16);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// second part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 20);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// third part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 20);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0.0001f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}
		}

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> pair = builder.buildWithTrainer();

		NeuralNetworkImpl nn = (NeuralNetworkImpl) pair.getRight().getNeuralNetwork();
		Layer input = nn.getInputLayer();

		assertEquals(1, input.getConnections().size());

		Conv2DConnection conv1c = (Conv2DConnection) input.getConnections().get(0);
		assertEquals(3, conv1c.getInputFilters());
		assertEquals(16, conv1c.getOutputFilters());
		assertEquals(5, conv1c.getFilterRows());
		assertEquals(5, conv1c.getFilterColumns());
		assertEquals(32, conv1c.getInputFeatureMapRows());
		assertEquals(32, conv1c.getInputFeatureMapColumns());
		assertEquals(1, conv1c.getRowStride());
		assertEquals(1, conv1c.getColumnStride());
		assertEquals(2, conv1c.getOutputColumnPadding());
		assertEquals(2, conv1c.getOutputRowPadding());
		assertEquals(32, conv1c.getOutputFeatureMapRowsWithPadding());
		assertEquals(32, conv1c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(28 * 28 * 16, conv1c.getOutputUnitCount());

		Conv2DConnection conv1b = (Conv2DConnection) conv1c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv1b.getInputFilters());
		assertEquals(16, conv1b.getOutputFilters());
		assertEquals(1, conv1b.getFilterRows());
		assertEquals(1, conv1b.getFilterColumns());
		assertEquals(28, conv1b.getInputFeatureMapRows());
		assertEquals(28, conv1b.getInputFeatureMapColumns());
		assertEquals(1, conv1b.getRowStride());
		assertEquals(1, conv1b.getColumnStride());
		assertEquals(2, conv1b.getOutputColumnPadding());
		assertEquals(2, conv1b.getOutputRowPadding());
		assertEquals(28 * 28 * 16, conv1b.getOutputUnitCount());
		assertEquals(32 * 32 * 16, conv1b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool1 = (Subsampling2DConnection) conv1c.getOutputLayer().getConnections().get(2);
		assertEquals(16, pool1.getFilters());
		assertEquals(2, pool1.getColumnStride());
		assertEquals(2, pool1.getRowStride());
		assertEquals(16, pool1.getOutputFeatureMapRows());
		assertEquals(16, pool1.getOutputFeatureMapColumns());
		assertEquals(16, pool1.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, pool1.getOutputFeatureMapColumnsWithPadding());
		assertEquals(32 * 32 * 16, pool1.getInputUnitCount());
		assertEquals(16 * 16 * 16, pool1.getOutputUnitCount());

		Conv2DConnection conv2c = (Conv2DConnection) pool1.getOutputLayer().getConnections().get(1);
		assertEquals(16, conv2c.getInputFilters());
		assertEquals(20, conv2c.getOutputFilters());
		assertEquals(5, conv2c.getFilterRows());
		assertEquals(5, conv2c.getFilterColumns());
		assertEquals(16, conv2c.getInputFeatureMapRows());
		assertEquals(16, conv2c.getInputFeatureMapColumns());
		assertEquals(1, conv2c.getRowStride());
		assertEquals(1, conv2c.getColumnStride());
		assertEquals(2, conv2c.getOutputColumnPadding());
		assertEquals(2, conv2c.getOutputRowPadding());
		assertEquals(12, conv2c.getOutputFeatureMapRows());
		assertEquals(12, conv2c.getOutputFeatureMapColumns());
		assertEquals(16, conv2c.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, conv2c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(12 * 12 * 20, conv2c.getOutputUnitCount());
		assertEquals(16 * 16 * 20, conv2c.getOutputUnitCountWithPadding());

		Conv2DConnection conv2b = (Conv2DConnection) conv2c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv2b.getInputFilters());
		assertEquals(20, conv2b.getOutputFilters());
		assertEquals(1, conv2b.getFilterRows());
		assertEquals(1, conv2b.getFilterColumns());
		assertEquals(12, conv2b.getInputFeatureMapRows());
		assertEquals(12, conv2b.getInputFeatureMapColumns());
		assertEquals(1, conv2b.getRowStride());
		assertEquals(1, conv2b.getColumnStride());
		assertEquals(2, conv2b.getOutputColumnPadding());
		assertEquals(2, conv2b.getOutputRowPadding());
		assertEquals(12, conv2b.getOutputFeatureMapRows());
		assertEquals(12, conv2b.getOutputFeatureMapColumns());
		assertEquals(16, conv2b.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, conv2b.getOutputFeatureMapColumnsWithPadding());
		assertEquals(12 * 12 * 20, conv2b.getOutputUnitCount());
		assertEquals(16 * 16 * 20, conv2b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool2 = (Subsampling2DConnection) conv2c.getOutputLayer().getConnections().get(2);
		assertEquals(20, pool2.getFilters());
		assertEquals(2, pool2.getColumnStride());
		assertEquals(2, pool2.getRowStride());
		assertEquals(8, pool2.getOutputFeatureMapRows());
		assertEquals(8, pool2.getOutputFeatureMapColumns());
		assertEquals(8, pool2.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, pool2.getOutputFeatureMapColumnsWithPadding());
		assertEquals(16 * 16 * 20, pool2.getInputUnitCount());
		assertEquals(8 * 8 * 20, pool2.getOutputUnitCount());

		Conv2DConnection conv3c = (Conv2DConnection) pool2.getOutputLayer().getConnections().get(1);
		assertEquals(20, conv3c.getInputFilters());
		assertEquals(20, conv3c.getOutputFilters());
		assertEquals(5, conv3c.getFilterRows());
		assertEquals(5, conv3c.getFilterColumns());
		assertEquals(8, conv3c.getInputFeatureMapRows());
		assertEquals(8, conv3c.getInputFeatureMapColumns());
		assertEquals(1, conv3c.getRowStride());
		assertEquals(1, conv3c.getColumnStride());
		assertEquals(2, conv3c.getOutputColumnPadding());
		assertEquals(2, conv3c.getOutputRowPadding());
		assertEquals(4, conv3c.getOutputFeatureMapRows());
		assertEquals(4, conv3c.getOutputFeatureMapColumns());
		assertEquals(8, conv3c.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, conv3c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(4 * 4 * 20, conv3c.getOutputUnitCount());
		assertEquals(8 * 8 * 20, conv3c.getOutputUnitCountWithPadding());

		Conv2DConnection conv3b = (Conv2DConnection) conv3c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv3b.getInputFilters());
		assertEquals(20, conv3b.getOutputFilters());
		assertEquals(1, conv3b.getFilterRows());
		assertEquals(1, conv3b.getFilterColumns());
		assertEquals(4, conv3b.getInputFeatureMapRows());
		assertEquals(4, conv3b.getInputFeatureMapColumns());
		assertEquals(1, conv3b.getRowStride());
		assertEquals(1, conv3b.getColumnStride());
		assertEquals(2, conv3b.getOutputColumnPadding());
		assertEquals(2, conv3b.getOutputRowPadding());
		assertEquals(4, conv3b.getOutputFeatureMapRows());
		assertEquals(4, conv3b.getOutputFeatureMapColumns());
		assertEquals(8, conv3b.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, conv3b.getOutputFeatureMapColumnsWithPadding());
		assertEquals(4 * 4 * 20, conv3b.getOutputUnitCount());
		assertEquals(8 * 8 * 20, conv3b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool3 = (Subsampling2DConnection) conv3c.getOutputLayer().getConnections().get(2);
		assertEquals(20, pool3.getFilters());
		assertEquals(2, pool3.getColumnStride());
		assertEquals(2, pool3.getRowStride());
		assertEquals(4, pool3.getOutputFeatureMapRows());
		assertEquals(4, pool3.getOutputFeatureMapColumns());
		assertEquals(4, pool3.getOutputFeatureMapRowsWithPadding());
		assertEquals(4, pool3.getOutputFeatureMapColumnsWithPadding());
		assertEquals(8 * 8 * 20, pool3.getInputUnitCount());
		assertEquals(4 * 4 * 20, pool3.getOutputUnitCount());

		FullyConnected fc1 = (FullyConnected) pool3.getOutputLayer().getConnections().get(1);
		assertEquals(4 * 4 * 20, fc1.getInputUnitCount());
		assertEquals(10, fc1.getOutputUnitCount());
	}

}