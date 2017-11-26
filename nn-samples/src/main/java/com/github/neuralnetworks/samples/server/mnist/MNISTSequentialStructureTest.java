package com.github.neuralnetworks.samples.server.mnist;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.MultipleNeuronsSimpleOutputError;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * @author Urko
 *         Class that test structure by structure CNNs on MNIST dataset. Useful to detect failing kernels
 */
public class MNISTSequentialStructureTest extends JUnitOnServerStarter
{
	private static final Logger logger = LoggerFactory.getLogger(MNISTSequentialStructureTest.class);
	private Random r = new Random(123);
	private byte[] seed = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	
	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		// Configuration with precompilation
		RuntimeConfiguration conf3 = new RuntimeConfiguration();
		conf3.setCalculationProvider(CalculationProvider.OPENCL);
		conf3.setUseDataSharedMemory(false);
		conf3.setUseWeightsSharedMemory(false);
		conf3.getOpenCLConfiguration().setAggregateOperations(true);
		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf3.getOpenCLConfiguration().setUseOptionsString(true);
		conf3.getOpenCLConfiguration().setRestartLibraryAfterPhase(true);
		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf3 });

		// Configuration without precompilation
		RuntimeConfiguration conf4 = new RuntimeConfiguration();
		conf4.setCalculationProvider(CalculationProvider.OPENCL);
		conf4.setUseDataSharedMemory(false);
		conf4.setUseWeightsSharedMemory(false);
		conf4.getOpenCLConfiguration().setAggregateOperations(true);
		conf4.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf4.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf4.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf4.getOpenCLConfiguration().setUseOptionsString(false);
		conf4.getOpenCLConfiguration().setRestartLibraryAfterPhase(true);
		conf4.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf4 });

		return configurations;
	}


	/**
	 * Structure : 1FC
	 */
	public float structure1FC(int epochs)
	{
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 784, 10 }, false);
		mlp.setLayerCalculator(CalculationFactory.lcSigmoid(mlp, OperationsFactory.sigmoidFunction()));

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsSimpleOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(seed, -0.01f, 0.01f)), 0.5f, 0.9f, 0.001f, 0f, 0f, 10, 1000, epochs);

		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogWeights(true);
		ls.setLogBatchLoss(false);
		ls.setLogInterval(1000);
		bpt.addEventListener(ls);

		bpt.train();
		bpt.test();

		return bpt.getOutputError().getTotalNetworkError();
	}
	

	/**
	 * Structure : 1FC
	 */
	public float structure1FC2(int epochs)
	{
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 784, 10 }, true);
		mlp.setLayerCalculator(CalculationFactory.lcRelu(mlp, OperationsFactory.softmaxFunction()));

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsSimpleOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(seed, -0.01f, 0.01f)), 0.05f, 0.5f, 0f, 0f, 0f, 10, 1000, epochs);

		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(false);
		ls.setLogInterval(1000);
		bpt.addEventListener(ls);

		RandomInitializerImpl ri = new RandomInitializerImpl(new Random(1234));
		bpt.getNeuralNetwork().getConnections().forEach(c -> ri.initialize(((WeightsConnections) c).getWeights()));
		//bpt.train();
		bpt.test();

		return bpt.getOutputError().getTotalNetworkError();
	}



	/**
	 * Structure : 2FC
	 */
	public float structure2FC(int epochs)
	{
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 784, 300, 10 }, true);
		mlp.setLayerCalculator(CalculationFactory.lcRelu(mlp, OperationsFactory.softmaxFunction()));

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsSimpleOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(seed, -0.01f, 0.01f)), 0.05f, 0.5f, 0f, 0f, 0f, 10, 1000, epochs);

		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

		bpt.train();
		bpt.test();

		return bpt.getOutputError().getTotalNetworkError();
	}

	/**
	 * Structure : 1CONV + 1FC
	 */
	public float structure1FC1Conv(int epochs)
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 6);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
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
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(100);
				builder.setTestBatchSize(1000);
			}
		}

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

		// training
		bpt.train();

		// testing
		bpt.test();
		
		return bpt.getOutputError().getTotalNetworkError();
	}


	/**
	 * Structure : 1CONV + pooling + 1FC
	 */
	public float structure1ConvPool1FC(int epochs)
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 6);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(100);
				builder.setTestBatchSize(1000);
			}
		}

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

		// training
		bpt.train();

		// testing
		bpt.test();
		
		return bpt.getOutputError().getTotalNetworkError();
	}


	/**
	 * Structure : 2CONV + 1FC
	 */
	public float structure2Conv1FC(int epochs)
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		
		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 6);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.07f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.1f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 14);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.07f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.1f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(100);
				builder.setTestBatchSize(1000);
			}
		}

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

		// training
		bpt.train();

		// testing
		bpt.test();
		
		return bpt.getOutputError().getTotalNetworkError();
	}


	/**
	 * Structure : 2CONV + Pooling + 1FC
	 */
	public float structure2convPool1FC(int epochs)
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 6);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.07f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.1f);
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
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 14);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.07f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.1f);
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
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(seed, 0f, 0.01f)));

				builder.setLearningRate(0.001f);
				builder.setMomentum(0.9f);
				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(100);
				builder.setTestBatchSize(1000);
			}
		}

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

		// training
		bpt.train();

		// testing
		bpt.test();
		
		return bpt.getOutputError().getTotalNetworkError();
	}


	public void jUnitTest(int epochs, String folder)
	{
		
//		float onefcRandom = structure1FC2(epochs);
		float onefc = structure1FC(epochs);
//		float twofc = structure2FC(epochs);
//		float onefcOneConv = structure1FC1Conv(epochs);
//		float onefcTwoConv = structure2Conv1FC(epochs);
//		float onefcPoolOneConv = structure1ConvPool1FC(epochs);
//		float onefcPoolTwoConv = structure2convPool1FC(epochs);
//		
		logger.info("Test error 1FC: " + onefc);
//		logger.info("Test error 1FCRandom: " + onefcRandom);
//		logger.info("Test error 2FC: " + twofc);
//		logger.info("Test error 1FC1Conv: " + onefcOneConv);
//		logger.info("Test error 1FC2Conv: " + onefcTwoConv);
//		logger.info("Test error 1FC1ConvPool: " + onefcPoolOneConv);
//		logger.info("Test error 1FC2ConvPool: " + onefcPoolTwoConv);

	}

	public static void main(String[] args)
	{

		JUnitOnServerStarter jUnitOnServerStarter = new MNISTSequentialStructureTest();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
