package com.github.neuralnetworks.samples.cifar;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.input.image.ImageResizeStrategy;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * a special test for vadim with the cifar images, 1 epoch, relu, 3 conv, 3 pooling(1 max, 2 avg), 2 fully connections
 * 
 * @author tmey
 */
public class VadimsTest
{

	private static RuntimeConfiguration configureGlobalRuntimeEnvironment()
	{
		RuntimeConfiguration conf = new RuntimeConfiguration();

		conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		conf.setUseDataSharedMemory(false);
		conf.setUseWeightsSharedMemory(false);
		conf.getOpenCLConfiguration().setAggregateOperations(true);
		conf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

		Environment.getInstance().setRuntimeConfiguration(conf);
		return conf;
	}

	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
			, com.github.neuralnetworks.training.TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider)
	{
		// environment

		configureGlobalRuntimeEnvironment();

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		// network

		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", imageSize, imageSize, nrOfInputFeatures));

			// first part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// second part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// third part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			builder.addLayerBuilder(new FullyConnectedLayerBuilder(64));

			builder.addLayerBuilder(new FullyConnectedLayerBuilder(numberOfCategories));
		}

		// trainer

		{
			builder.setTrainingSet(trainSetProvider);
			builder.setTestingSet(testSetProvider);

			builder.setLearningRate(0.001f);
			builder.setMomentum(0.9f);

			builder.setEpochs(1);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(128);
		}

		return builder.buildWithTrainer();

	}

	public void test()
	{

		// proprocessing of the images
		CIFARInputProvider.CIFAR10TrainingInputProvider cifarTrainInputProvider = new CIFARInputProvider.CIFAR10TrainingInputProvider("cifar-10-batches-bin");
		cifarTrainInputProvider.getProperties().setGroupByChannel(true);
		cifarTrainInputProvider.getProperties().setScaleColors(true);
		cifarTrainInputProvider.getProperties().setSubtractMean(true);
		cifarTrainInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 24, true));
		cifarTrainInputProvider.reset();
		Util.inputToFloat(cifarTrainInputProvider, "cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float");

		CIFARInputProvider.CIFAR10TestingInputProvider cifarTestInputProvider = new CIFARInputProvider.CIFAR10TestingInputProvider("cifar-10-batches-bin");
		cifarTestInputProvider.getProperties().setGroupByChannel(true);
		cifarTestInputProvider.getProperties().setScaleColors(true);
		cifarTestInputProvider.getProperties().setSubtractMean(true);
		cifarTestInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 24, true));
		cifarTestInputProvider.reset();
		Util.inputToFloat(cifarTestInputProvider, "cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float");


		// create input provider

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 24 * 24, 10, 50000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 24 * 24, 10, 10000);


		// create neural network and trainer

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);

		Trainer<NeuralNetwork> trainer = neuralNetworkAndTrainer.getRight();

		trainer.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));


		// start training

		trainer.train();
	}

	public static void main(String... args)
	{
		VadimsTest vadimsTest = new VadimsTest();
		vadimsTest.test();
	}

}
