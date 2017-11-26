package com.github.neuralnetworks.samples.server.cifar;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.input.MultipleNeuronsSimpleOutputError;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * @author Urko
 */
public class CifarServer1FCTest extends JUnitOnServerStarter
{

	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

		RuntimeConfiguration conf3 = new RuntimeConfiguration();
		conf3.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		conf3.setUseDataSharedMemory(false);
		conf3.setUseWeightsSharedMemory(false);
		conf3.getOpenCLConfiguration().setAggregateOperations(true);
		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf3.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf3 });

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
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));


			// fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setLearningRate(0.001f);
				fullyConnectedLayerBuilder.setL1weightDecay(0.03f);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasL1weightDecay(0);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}


			// trainer
			{

//				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
//				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);

				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 60000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 32 * 32, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);
				builder.setError(new MultipleNeuronsSimpleOutputError());
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

//
//		logger.info("Configuration: " + "\n"
//				+ "Layers : 1 Conv + 2 FC " + "\n"
//
//				+ "\tInput: " + "\n"
//				+ "\t\tImage size: 28x28" + "\n"
//				+ "\t\tNumber Channels: 1" + "\n"
//
//				+ "\tConv(1): " + "\n"
//				+ "\t\tPadding: 0" + "\n"
//				+ "\t\tStride: 1" + "\n"
//				+ "\t\tPatch: 5x5" + "\n"
//				+ "\t\tFM: 6" + "\n"
//				+ "\t\tLearning Rate: 0.01" + "\n"
//				+ "\t\tMomentum: 0.9" + "\n"
//				+ "\t\tActivation: ReLu" + "\n"
//
//				+ "\tPool(1): " + "\n"
//				+ "\t\tSize: 2" + "\n"
//				+ "\t\tStride: 2" + "\n"
//				+ "\t\tType: Max" + "\n"
//				+ "\t\tFM: 6" + "\n"
//
//				+ "\tFC(1): " + "\n"
//				+ "\t\tNeurons: 300" + "\n"
//				+ "\t\tLearning Rate: 0.01" + "\n"
//				+ "\t\tMomentum: 0.9" + "\n"
//				+ "\t\tActivation: ReLu" + "\n"
//
//				+ "\tFC(2): " + "\n"
//				+ "\t\tNeurons: 10" + "\n"
//				+ "\t\tLearning Rate: 0.01" + "\n"
//				+ "\t\tMomentum: 0.9" + "\n"
//				+ "\t\tActivation: SoftMax" + "\n"
//
//				+ "Training : " + "\n"
//				+ "\tNumber of images: 60k" + "\n"
//				+ "\tLearning Rate: 0.01" + "\n"
//				+ "\tMomentum: 0.9" + "\n"
//				+ "\tEpochs: " + epochs + "\n"
//				+ "\tBatch size: 100" + "\n"
//
//				+ "Test : " + "\n"
//				+ "\tNumber of images: 10k" + "\n"
//				+ "\tBatch size: 1000" + "\n"
//
//				);

		if (bpt.getOutputError().getTotalNetworkError() > 0.1)
		{
			throw new IllegalStateException("error was to high! " + bpt.getOutputError().getTotalNetworkError() + " > 0.1");
		}


	}

	public static void main(String[] args)
	{
		JUnitOnServerStarter jUnitOnServerStarter = new CifarServer1FCTest();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
