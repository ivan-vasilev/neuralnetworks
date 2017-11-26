package com.github.neuralnetworks.samples.cifar;

import java.io.File;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
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
 *
 * test only parts of the hole network
 *
 * a network based on cudas network for cifar that trains in 80 seconds https://code.google.com/p/cuda-convnet/<br>
 * network definition: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg<br>
 * hyperparameter: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-80sec.cfg
 * 
 * @author tmey
 */
public class StepByStepTest
{

	private static final Logger logger = LoggerFactory.getLogger(StepByStepTest.class);

	private TrainingInputProvider trainInputProvider;
	private TrainingInputProvider testInputProvider;


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

	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getFullNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
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

				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

		}

		// trainer

		{
			builder.setTrainingSet(trainSetProvider);
			builder.setTestingSet(testSetProvider);

			builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(0.00001f, 0.1f)));
//			builder.setRand(new NNRandomInitializer(new DummyFixedInitializer()));
//            builder.setRand(new NNRandomInitializer(new RandomInitializerImpl(new Random(123456789),0,0.01f)));

			builder.setLearningRate(0.001f);
			builder.setMomentum(0.9f);
			builder.setL2weightDecay(0.004f / 128f);
//            builder.setL1weightDecay(0.004f);

			builder.setEpochs(3);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
	}


	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getOneFullyConnectedNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
			, TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider)
	{
		// environment

		configureGlobalRuntimeEnvironment();

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", imageSize, imageSize, nrOfInputFeatures));


			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

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

			builder.setEpochs(3);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
	}

	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getTwoFullyConnectedNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
			, TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider)
	{
		// environment

		configureGlobalRuntimeEnvironment();

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", imageSize, imageSize, nrOfInputFeatures));


			// first fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(64);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

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

			builder.setEpochs(3);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
	}

	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getConvAndFullyConnectedNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
			, TrainingInputProvider trainSetProvider, TrainingInputProvider testSetProvider, int padding)
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
				convolutionalLayerBuilder.setPaddingSize(padding);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.0001));
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0);

				builder.addLayerBuilder(convolutionalLayerBuilder);
			}


			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

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

			builder.setEpochs(3);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
	}

	private static Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getConvAndPoolingAndFullyConnectedNeuralNetworkAndTrainer(int imageSize, int nrOfInputFeatures, int numberOfCategories
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
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.0001));
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0);

				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}


			// second fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(numberOfCategories);
				fullyConnectedLayerBuilder.setL2weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new NormalDistributionInitializer(0, 0.1));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0);

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

			builder.setEpochs(3);
			builder.setTrainingBatchSize(128);
			builder.setTestBatchSize(512);
		}
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();

		return neuralNetworkTrainerPair;
	}

	private void prepareTestData()
	{

		logger.info("prepare train and test data...");
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

		trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 24 * 24, 10, 50000);
		testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 24 * 24, 10, 10000);
	}

	public void test()
	{

		// proprocessing of the images
		prepareTestData();


		// test every configuration
		{
			logger.info("one fully");
			Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getOneFullyConnectedNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);
			testNetwork(neuralNetworkAndTrainer);
		}

		{
			logger.info("two fully");
			Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getTwoFullyConnectedNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);
			testNetwork(neuralNetworkAndTrainer);
		}

		for (int i = 0; i <= 2; i++)
		{
			logger.info("convolutional(padding " + i + ") + fully");
			Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getConvAndFullyConnectedNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider, i);
			testNetwork(neuralNetworkAndTrainer);
		}

		{
			logger.info("convolutional + pooling + fully");
			Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getConvAndPoolingAndFullyConnectedNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);
			testNetwork(neuralNetworkAndTrainer);
		}

		{
			logger.info("everything");
			Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getFullNeuralNetworkAndTrainer(24, 3, 10, trainInputProvider, testInputProvider);
			testNetwork(neuralNetworkAndTrainer);
		}
	}

	private void testNetwork(Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer)
	{
		Trainer<NeuralNetwork> trainer = neuralNetworkAndTrainer.getRight();

		trainer.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
		EarlySynchronizeEventListener earlySynchronizeEventListener = new EarlySynchronizeEventListener(trainer);
		earlySynchronizeEventListener.setSaveFile(new File("testNetwork.cnn"));
//		earlySynchronizeEventListener.setSampleStep(1);
		trainer.addEventListener(earlySynchronizeEventListener);

		// start training
		ValuesProvider results = TensorFactory.tensorProvider(trainer.getNeuralNetwork(), 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor targetTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getOutputLayer()).getDimensions());
		Tensor inputTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getInputLayer()).getDimensions());
		testInputProvider.getNextInput(inputTensor);
		testInputProvider.getNextTarget(targetTensor);

//		new LearnTester().testIfNetworkLearns(trainer, 1, inputTensor, targetTensor);

		trainer.train();

		String result = ConnectionAnalysis.analyseConnectionWeights(neuralNetworkAndTrainer.getLeft());

		System.out.println("" + result);

		if (result.contains("NaN"))
		{
			throw new IllegalStateException("Infinite number problem!");
		}

//        trainer.test();
	}

	public static void main(String... args)
	{
		StepByStepTest cudaTest = new StepByStepTest();
		cudaTest.test();
	}

}
