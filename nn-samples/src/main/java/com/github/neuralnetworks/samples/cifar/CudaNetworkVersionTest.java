package com.github.neuralnetworks.samples.cifar;

import java.io.File;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.analyzing.ConnectionAnalysis;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.input.image.ImageResizeStrategy;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.EarlySynchronizeEventListener;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.NormalDistributionInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * a network based on cudas network for cifar that trains in 80 seconds https://code.google.com/p/cuda-convnet/<br>
 * network definition: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg<br>
 * hyperparameter: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-80sec.cfg
 * 
 * @author tmey
 */
public class CudaNetworkVersionTest
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
			, TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider)
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
				convolutionalLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.0001));
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0);
//                convolutionalLayerBuilder.setActivationType(ActivationType.Sigmoid);

				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// second part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.01));
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0);
//                convolutionalLayerBuilder.setActivationType(ActivationType.Sigmoid);

				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// third part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.01));
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0);
//                convolutionalLayerBuilder.setActivationType(ActivationType.Sigmoid);

				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// first fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(64);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);
//                fullyConnectedLayerBuilder.setActivationType(ActivationType.Sigmoid);

				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);
//                fullyConnectedLayerBuilder.setActivationType(ActivationType.Sigmoid);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

		}

		// trainer

		{
			builder.setTrainingSet(trainSetProvider);
			builder.setTestingSet(testSetProvider);

			builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(0.00001f, 0.1f)));

			builder.setLearningRate(0.001f);
			builder.setMomentum(0.9f);
			builder.setL2weightDecay(0.004f / 128f);
//            builder.setL1weightDecay(0.004f);

			builder.setEpochs(5);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
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
		cifarTrainInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 24, true));
		cifarTestInputProvider.reset();
		Util.inputToFloat(cifarTestInputProvider, "cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float");


		// create input provider

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 24 * 24, 10, 50000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 24 * 24, 10, 10000);


		// create neural network and trainer

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);

		Trainer<NeuralNetwork> trainer = neuralNetworkAndTrainer.getRight();

		trainer.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
		EarlySynchronizeEventListener earlySynchronizeEventListener = new EarlySynchronizeEventListener(trainer);
		earlySynchronizeEventListener.setSaveFile(new File("testNetwork.cnn"));
		earlySynchronizeEventListener.setSampleStep(1);
		trainer.addEventListener(earlySynchronizeEventListener);

		// start training

		trainer.train();

		System.out.println("" + ConnectionAnalysis.analyseConnectionWeights(neuralNetworkAndTrainer.getLeft()));

		trainer.test();
	}

	public static void main(String... args)
	{
		CudaNetworkVersionTest cudaTest = new CudaNetworkVersionTest();
		cudaTest.test();
	}

}
