package com.github.neuralnetworks.analyzing.network.comparison;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.analyzing.network.comparison.CPUAndGPUComparison;
import com.github.neuralnetworks.analyzing.network.comparison.DifferentNetworksException;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.input.MultipleNeuronsSimpleOutputError;
import com.github.neuralnetworks.input.RandomInputProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;

public class CifarCPUAndGPUComparisonTest extends AbstractTest
{
	private static final Logger logger = LoggerFactory.getLogger(CifarCPUAndGPUComparisonTest.class);

	private TrainingInputProvider testInputProvider;

	public CifarCPUAndGPUComparisonTest()
	{
		// create input provider
		testInputProvider = new RandomInputProvider(3, 32, 10, new Random(123));// new SimpleInputProvider(new float[][] { input }, new float[][] { target });
	}

//	@After
//	public void after(){
//		OpenCLCore.getInstance().finalizeDeviceAll();
//	}

	@Test
	public void testFc()
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();


		long seed = 123;

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));


			// fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setLearningRate(0.001f);
				fullyConnectedLayerBuilder.setL1weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0f, 0.01f, seed));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL1weightDecay(0);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}


			// trainer
			{
				builder.setTrainingSet(testInputProvider);
				builder.setTestingSet(testInputProvider);
				builder.setError(new MultipleNeuronsSimpleOutputError());
				builder.setRand(new NNRandomInitializer(new RandomInitializerImpl(0f, 0.01f, seed)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(1);
				builder.setTrainingBatchSize(10);
				builder.setTestBatchSize(1000);
			}
		}

		CPUAndGPUComparison cpuAndGPUComparison = new CPUAndGPUComparison();
		cpuAndGPUComparison.getComparison().getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(new File("CifarCPUAndGPUComparison" + File.separator + "testFC" + File.separator));

		try
		{
			cpuAndGPUComparison.compare(builder);
		} catch (DifferentNetworksException e)
		{
			logger.error("", e);
			assertTrue(e.getMessage(), false);
		}
	}

	@Test
	public void testConvFC()
	{

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		long seed = 123;
		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 30);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0f, 0.01f, seed));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);
			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(01f, 0.01f, seed));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{

				builder.setTrainingSet(testInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setRand(new NNRandomInitializer(new RandomInitializerImpl(0f, 0.01f, seed)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(1);
				builder.setTrainingBatchSize(10);
				builder.setTestBatchSize(1000);
			}
		}

		CPUAndGPUComparison cpuAndGPUComparison = new CPUAndGPUComparison();
		cpuAndGPUComparison.getComparison().getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(new File("CifarCPUAndGPUComparison" + File.separator + "testConfFC" + File.separator));

		try
		{
			cpuAndGPUComparison.compare(builder);
		} catch (DifferentNetworksException e)
		{
			logger.error("", e);
			assertTrue(e.getMessage(), false);
		}
	}

	@Test
	public void testConvPoolFC()
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		long seed = 123456789;

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 4));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0, 0.01f, seed));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(0f, 0.01f, seed));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				// convolutionalLayerBuilder.setL1weightDecay(0.01f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0, 0.01f, seed));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(0f, 0.01f, seed));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				// convolutionalLayerBuilder.setL1weightDecay(0.01f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(3, 64);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0, 0.01f, seed));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(0f, 0.01f, seed));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				// convolutionalLayerBuilder.setL1weightDecay(0.01f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.ReLU);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(0, 0.01f, seed));
				fullyConnectedLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(0, 0.01f, seed));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				// fullyConnectedLayerBuilder.setL1weightDecay(0.01f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				builder.setTrainingSet(testInputProvider);
				builder.setTestingSet(testInputProvider);

				// builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				// builder.setLearningRate(0.001f);
				// builder.setMomentum(0.9f);
				builder.setEpochs(1);
				builder.setTrainingBatchSize(10);
				builder.setTestBatchSize(128);
			}
		}

		CPUAndGPUComparison cpuAndGPUComparison = new CPUAndGPUComparison();
		cpuAndGPUComparison.getComparison().getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(new File("CifarCPUAndGPUComparison" + File.separator + "testConfPoolFC" + File.separator));
		cpuAndGPUComparison.getComparison().getSimilarNetworkWeightsComparison().setMaxDifference(0.00001f);


		try
		{
			cpuAndGPUComparison.compare(builder);
		} catch (DifferentNetworksException e)
		{
			logger.error("", e);
			assertTrue(e.getMessage(), false);
		}
	}

}