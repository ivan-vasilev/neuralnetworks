package com.github.neuralnetworks.samples.server.mnist;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
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
 *         Class that mimics the network structure posted here https://leipzig.exb.de/confluence/display/RESNLP/MNIST+TESTS
 *         The 3 integer arguments are needed for running the main: arg[0] = #epochs, arg[1] = configuration, arg[2] = device
 *         Configuration: LeNet  3 Fully connected +2 Convolutional + max pooling 
 *         3.05% Test error 1 epoch
 */
public class LeNetTest extends JUnitOnServerStarter
{


	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();
//
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

	@Override
	public void jUnitTest(int epochs, String folder)
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		Random r = new Random(123);
		byte[] seed = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 20);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.02f);
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
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 50);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.02f);
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
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(500);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.02f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.1f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.1f);
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

				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(64);
				builder.setTestBatchSize(1000);
			}
		}
			BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

			// log data
			LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
			ls.setLogBatchLoss(true);
			// ls.setLogWeightUpdates(true);
			ls.setLogInterval(5000);
			bpt.addEventListener(ls);

			// training
			bpt.train();

			// testing
			bpt.test();


		if (bpt.getOutputError().getTotalNetworkError() > 0.1)
		{
			throw new IllegalStateException("error was to high! " + bpt.getOutputError().getTotalNetworkError() + " > 0.1");
		}


	}

	public static void main(String[] args)
	{
		JUnitOnServerStarter jUnitOnServerStarter = new LeNetTest();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
