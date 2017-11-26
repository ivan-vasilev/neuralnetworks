package com.github.neuralnetworks.samples.server.cifar;

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
 */
public class CifarConvNetTest extends JUnitOnServerStarter
{


	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.CPU);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

		RuntimeConfiguration conf3 = new RuntimeConfiguration();
		conf3.setCalculationProvider(CalculationProvider.OPENCL);
		conf3.setUseDataSharedMemory(false);
		conf3.setUseWeightsSharedMemory(false);
		conf3.getOpenCLConfiguration().setAggregateOperations(true);
		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
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

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.0001f));
				convolutionalLayerBuilder.setLearningRate(0.001f);
				convolutionalLayerBuilder.setMomentum(0.9f);
//				 convolutionalLayerBuilder.setL1weightDecay(0.004f);
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
//				 convolutionalLayerBuilder.setBiasL1weightDecay(0.004f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.ReLU);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.001f);
				convolutionalLayerBuilder.setMomentum(0.9f);
//				 convolutionalLayerBuilder.setL1weightDecay(0.004f);
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
//				 convolutionalLayerBuilder.setBiasL1weightDecay(0.004f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setStrideSize(1);
				
				convolutionalLayerBuilder.setLearningRate(0.001f);
				convolutionalLayerBuilder.setMomentum(0.9f);
//				 convolutionalLayerBuilder.setL1weightDecay(1f);
				convolutionalLayerBuilder.setBiasLearningRate(0.002f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
//				 convolutionalLayerBuilder.setBiasL1weightDecay(1f);
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
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.001f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				// fullyConnectedLayerBuilder.setL1weightDecay(0.01f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.002f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				// fullyConnectedLayerBuilder.setBiasL1weightDecay(0.01f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 50000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 32 * 32, 10, 10000);
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
		// ls.setLogWeightUpdates(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

//		EarlyStoppingListener es2 = new EarlyStoppingListener(bpt.getTestingInputProvider(), 0, "Test");
//		es2.setMiniBatchSize(1000);
//		es2.setOutputFile(new File("TestError.txt"));
//		bpt.addEventListener(es2);
		
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
		JUnitOnServerStarter jUnitOnServerStarter = new CifarConvNetTest();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
