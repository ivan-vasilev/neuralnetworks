package com.github.neuralnetworks.performance.tests;

import java.io.File;
import java.util.Random;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;

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
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider;
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider.CIFAR10TrainingInputProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * @author tmey
 */
public class PerformanceTest {

	public enum Runtime {
		CPU, APARAPI, NATIVE;
	}

	private Runtime runtime = Runtime.CPU;

	public void setRuntime(Runtime runtime) {
		this.runtime = runtime;
	}

	private RuntimeConfiguration configureGlobalRuntimeEnvironment() {

		RuntimeConfiguration conf = new RuntimeConfiguration();

		if (this.runtime.equals(Runtime.CPU)) {
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.APARAPI);
			conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
		} else if (this.runtime.equals(Runtime.APARAPI)) {
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.APARAPI);
			conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.GPU);
		} else if (this.runtime.equals(Runtime.NATIVE)) {
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
			conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		}
		// conf.getOpenCLConfiguration().setPreferredDevice(0);
		conf.setUseDataSharedMemory(false);
		conf.setUseWeightsSharedMemory(false);
		conf.getOpenCLConfiguration().setAggregateOperations(true);
		conf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		Environment.getInstance().setRuntimeConfiguration(conf);
		return conf;
	}

	public Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> prepareCfar10Test(String path) {

		// proprocessing of the images
		System.out.println("Prepare training data...");
		CIFAR10TrainingInputProvider cifarTrainInputProvider = new CIFAR10TrainingInputProvider(path);
		cifarTrainInputProvider.getProperties().setGroupByChannel(true);
		cifarTrainInputProvider.getProperties().setScaleColors(true);
		cifarTrainInputProvider.getProperties().setSubtractMean(true);
		cifarTrainInputProvider.getProperties().setParallelPreprocessing(false);
		// cifarTrainInputProvider.getProperties().setResizeStrategy(new
		// ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE,
		// 32, true));
		cifarTrainInputProvider.reset();
		Util.inputToFloat(cifarTrainInputProvider, path + File.separator + "train-data.float",
				path + File.separator + "train-labels.float");

		System.out.println("Prepare test data...");
		CIFARInputProvider.CIFAR10TestingInputProvider cifarTestInputProvider = new CIFARInputProvider.CIFAR10TestingInputProvider(
				path);
		cifarTestInputProvider.getProperties().setGroupByChannel(true);
		cifarTestInputProvider.getProperties().setScaleColors(true);
		cifarTestInputProvider.getProperties().setSubtractMean(true);
		cifarTestInputProvider.getProperties().setResizeStrategy(
				new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, 32, true));
		cifarTestInputProvider.reset();
		Util.inputToFloat(cifarTestInputProvider, path + File.separator + "testCifar10-data.float",
				path + File.separator + "testCifar10-labels.float");

		// create input provider
		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider(
				path + File.separator + "train-data.float", path + File.separator + "train-labels.float", 32 * 32, 10,
				50000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider(
				path + File.separator + "testCifar10-data.float", path + File.separator + "testCifar10-labels.float",
				32 * 32, 10, 10000);

		// activate precomilation flags for cl files
		Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().setUseOptionsString(true);

		// create neural network and trainer
		System.out.println("Create network...");
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = getNeuralNetworkAndTrainerCifar(32, 3,
				10, trainInputProvider, testInputProvider);
		return neuralNetworkAndTrainer;

	}

	private Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> getNeuralNetworkAndTrainerCifar(int imageSize,
			int nrOfInputFeatures, int numberOfCategories,
			com.github.neuralnetworks.training.TrainingInputProvider trainSetProvider,
			TrainingInputProvider testSetProvider) {
		// environment

		configureGlobalRuntimeEnvironment();

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		Random r = new Random(123456789);

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", imageSize, imageSize, nrOfInputFeatures));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
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
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
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
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				convolutionalLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
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
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(
						numberOfCategories);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				fullyConnectedLayerBuilder.setBiasWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
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

				// builder.setRand(new NNRandomInitializer(new
				// MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				// builder.setLearningRate(0.001f);
				// builder.setMomentum(0.9f);
				builder.setEpochs(1);
				builder.setTrainingBatchSize(128);
				builder.setTestBatchSize(128);
			}
		}

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();
		return neuralNetworkTrainerPair;

	}

	public long testCifar10() {

		final File path = FileUtils.getFile("src", "test", "resources", "cifar10");

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkAndTrainer = prepareCfar10Test(
				path.getAbsolutePath());
		Trainer<NeuralNetwork> trainer = neuralNetworkAndTrainer.getRight();

		// // log data
		// LogTrainingListener ls = new
		// LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(),
		// false, true);
		// ls.setLogBatchLoss(true);
		// // ls.setLogWeightUpdates(true);
		// ls.setLogInterval(5000);
		// trainer.addEventListener(ls);

		PerformanceEventListener listener = new PerformanceEventListener();
		trainer.addEventListener(listener);

		// start training
		System.out.println("Start training...");
		executeTraining(trainer);
		long time = listener.getTrainingRunTimeMs();
		System.out.println(Util.networkWeights(trainer.getNeuralNetwork()));
		return time;
	}

	public void executeTraining(Trainer<NeuralNetwork> trainer) {
		trainer.train();
	}

	public static void main(String... args) {

		Options options = new Options();
		options.addOption("h", "help", false, "show this help message");
		options.addOption("r", "runtime", true,
				"defines the runtime, possible options are 'cpu'(default, execute on cpu), 'native'(execute on GPU) and 'aparapi'(execute on GPU using Aparapi)");
		// options.addOption("d","dataset",true,"defines the dataset, possible values
		// are 'mnist' and 'cifar'(default)");
		CommandLineParser parser = new BasicParser();

		String dataset = "cifar"; // use only cifar for the time being
		Runtime runtime = Runtime.CPU;

		try {
			CommandLine cmd = parser.parse(options, args);

			if (cmd.hasOption("h")) {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("[options]...", options);
				System.exit(0);
				return;
			}
			if (cmd.hasOption("r")) {
				String value = cmd.getOptionValue("r");

				if (value.equals("cpu")) {
					runtime = Runtime.CPU;
				} else if (value.equals("aparapi")) {
					runtime = Runtime.APARAPI;
				} else if (value.equals("native")) {
					runtime = Runtime.NATIVE;
				} else {
					System.out.println("Unexpected runtime paramater: '" + value + "'");
					System.exit(-1);
					return;
				}
			}

			if (cmd.hasOption("d")) {
				String value = cmd.getOptionValue("d");

				if (value.equals("mnist")) {
					dataset = value;
				} else if (value.equals("cifar")) {
					dataset = value;
				} else {
					System.out.println("Unexpected dataset paramater: '" + value + "'");
					System.exit(-1);
					return;
				}
			}

		} catch (ParseException e) {
			e.printStackTrace();
			return;
		}

		System.out.println("Run test on "
				+ (runtime.equals(Runtime.CPU) ? "cpu" : (runtime.equals(Runtime.NATIVE) ? "gpu" : "gpu (aparapi)"))
				+ " with cifar10 dataset");

		PerformanceTest testRunner = new PerformanceTest();
		testRunner.setRuntime(runtime);
		long time = -1;
		if (dataset.equals("cifar")) {
			time = testRunner.testCifar10();
		} else {
			return;
		}

		System.out.println("##teamcity[testStarted name='Performance.Test' ]");
		System.out.println("##teamcity[testFinished name='Performance.Test' duration='" + time + "']");
		System.out.println("##teamcity[buildStatus status='SUCCESS' text='train 1 epoc took " + time + " msec']");
		System.exit(0);
	}

}
