package com.github.neuralnetworks.samples.server.mnist;

import java.util.ArrayList;
import java.util.List;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
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
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * @author Urko
 *         Class that mimics the network structure posted here https://leipzig.exb.de/confluence/display/RESNLP/MNIST+TESTS
 *         The 3 integer arguments are needed for running the main: arg[0] = #epochs, arg[1] = configuration, arg[2] = device
 *         Configuration: 2FC (300 & 10 neurons) layers
 *         3.36% Test error 1 epoch
 */
public class Mnist2FC extends JUnitOnServerStarter
{


	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.CPU);
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

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf2.setUseDataSharedMemory(false);
		conf2.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf2 });

		return configurations;
	}

	public void jUnitTest(int epochs, String folder)
	{
		byte[] seed = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 784, 300, 10 }, true);
		mlp.setLayerCalculator(CalculationFactory.lcRelu(mlp, OperationsFactory.softmaxFunction()));

		TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
		TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsSimpleOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(seed, -0.01f, 0.01f)), 0.05f, 0.5f, 0f, 0f, 0f, 10, 1000, epochs);

		// Listener that informs about the loss function status
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(false);
		ls.setLogInterval(1000);
		bpt.addEventListener(ls);

		// Net training
		bpt.train();

		// Test
		bpt.test();

		if (bpt.getOutputError().getTotalNetworkError() > 0.1)
		{
			throw new IllegalStateException("error was to high! " + bpt.getOutputError().getTotalNetworkError() + " > 0.1");
		}


	}

	public static void main(String[] args)
	{

		JUnitOnServerStarter jUnitOnServerStarter = new Mnist2FC();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
