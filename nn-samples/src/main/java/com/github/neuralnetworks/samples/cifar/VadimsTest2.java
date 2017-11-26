package com.github.neuralnetworks.samples.cifar;

import com.amd.aparapi.Kernel;
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
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * a special test for vadim with the cifar images, 1 epoch, relu, 3 conv, 3 pooling(1 max, 2 avg), 2 fully connections
 *
 * @author tmey
 */
public class VadimsTest2
{

	private static final Logger logger = LoggerFactory.getLogger(VadimsTest2.class);

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

	@SuppressWarnings("unused")
	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
			, com.github.neuralnetworks.training.TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider)
	{
		// environment

		configureGlobalRuntimeEnvironment();

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		long seed = 123456789;

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", imageSize, imageSize, nrOfInputFeatures));

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
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
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
				builder.setTrainingSet(testSetProvider);
				builder.setTestingSet(testSetProvider);

				// builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				// builder.setLearningRate(0.001f);
				// builder.setMomentum(0.9f);
				builder.setEpochs(1);
				builder.setTrainingBatchSize(128);
				builder.setTestBatchSize(128);
			}
		}

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) neuralNetworkTrainerPair.getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		// ls.setLogWeightUpdates(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);


		return neuralNetworkTrainerPair;

	}

	public void test()
	{
		test(false, -1);
	}

	public void test(boolean activatePrecompilation, int deviceID)
	{

		// proprocessing of the images
		CIFARInputProvider.CIFAR10TrainingInputProvider cifarTrainInputProvider = new CIFARInputProvider.CIFAR10TrainingInputProvider("cifar-10-batches-bin");
		cifarTrainInputProvider.getProperties().setGroupByChannel(true);
		cifarTrainInputProvider.getProperties().setScaleColors(true);
		cifarTrainInputProvider.getProperties().setSubtractMean(true);
		cifarTrainInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 32, true));
		cifarTrainInputProvider.reset();
		Util.inputToFloat(cifarTrainInputProvider, "cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float");

		CIFARInputProvider.CIFAR10TestingInputProvider cifarTestInputProvider = new CIFARInputProvider.CIFAR10TestingInputProvider("cifar-10-batches-bin");
		cifarTestInputProvider.getProperties().setGroupByChannel(true);
		cifarTestInputProvider.getProperties().setScaleColors(true);
		cifarTestInputProvider.getProperties().setSubtractMean(true);
		cifarTestInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 32, true));
		cifarTestInputProvider.reset();
		Util.inputToFloat(cifarTestInputProvider, "cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float");


		// create input provider

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 50000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 32 * 32, 10, 10000);

		// activate precompilation
		Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().setUseOptionsString(activatePrecompilation);

		if (deviceID > -1)
		{
			Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().setPreferredDevice(deviceID);
		}

		// create neural network and trainer

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getNeuralNetworkAndTrainer(32, 3, 10, trainInputProvider, testInputProvider);

		Trainer<NeuralNetwork> trainer = neuralNetworkAndTrainer.getRight();

		trainer.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));


		// start training

		trainer.train();
	}

	public static void main(String... args)
	{

		String deviceIDFlag = "-deviceid=";
		String precompilationFlag = "-precompilation";

		boolean usePrecompilation = false;
		int deviceID = -1;

		for (String arg : args)
		{
			if (arg.equalsIgnoreCase(precompilationFlag))
			{
				usePrecompilation = true;
				logger.info("precompilation activated");
			}

			if (arg.startsWith(deviceIDFlag))
			{
				deviceID = Integer.parseInt(arg.substring(deviceIDFlag.length()));
				logger.info("preferred deviceId: " + deviceID);
			}
		}

		if (!usePrecompilation)
		{
			System.out.println("(Use \"" + precompilationFlag + "\" as flag to use the precompilation.)");
		}
		if (deviceID <= 0)
		{
			System.out.println("(Use \"" + deviceIDFlag + "X\" as flag to use a specific device.)");
		}


		VadimsTest2 vadimsTest = new VadimsTest2();
		vadimsTest.test(usePrecompilation, deviceID);
	}

}
