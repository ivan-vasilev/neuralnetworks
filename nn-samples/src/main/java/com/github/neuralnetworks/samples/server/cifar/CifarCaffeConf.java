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
import com.github.neuralnetworks.training.random.GaussianRandomInitializerImpl;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * @author Urko
 */
public class CifarCaffeConf extends JUnitOnServerStarter
{

	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

//		RuntimeConfiguration conf1 = new RuntimeConfiguration();
//		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.CPU);
//		conf1.setUseDataSharedMemory(false);
//		conf1.setUseWeightsSharedMemory(false);
//		configurations.add(new RuntimeConfiguration[] { conf1 });

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

		Random r = new Random();
		long seed = 123456;

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
//				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.0001f));
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (3 * 5 * 5 + 32 * 5 * 5))));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				// convolutionalLayerBuilder.setL1weightDecay(0.01f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
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
//			convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (32 * 5 * 5 + 32 * 5 * 5))));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				// convolutionalLayerBuilder.setL1weightDecay(0.01f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
//			convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (32 * 5 * 5 + 64 * 5 * 5))));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(64);
//				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.0001f));
				fullyConnectedLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (64 * 5 * 5 + 64))));
				fullyConnectedLayerBuilder.setLearningRate(0.05f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.05f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}
			
			// fc
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
//				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.0001f));
				fullyConnectedLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (64  + 10))));
				fullyConnectedLayerBuilder.setLearningRate(0.05f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.05f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 50000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 32 * 32, 10, 10000);
				builder.setTrainingSet(trainInputProvider);
				builder.setTestingSet(testInputProvider);

				builder.setEpochs(epochs);
				builder.setTrainingBatchSize(100);
				builder.setTestBatchSize(100);
			}
		}
		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
		ls.setLogBatchLoss(true);
		// ls.setLogWeightUpdates(true);
		ls.setLogInterval(5000);
		bpt.addEventListener(ls);

//		bpt.addEventListener(new ValidationListener(new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 10000))
//		{
//			private static final long serialVersionUID = 1L;
//
//			private float prevPrevError;
//			private int epoch;
//
//			@Override
//			public void handleEvent(TrainingEvent event)
//			{
//				super.handleEvent(event);
//				if (event instanceof EpochFinishedEvent)
//				{
//					epoch++;
//					// if (prevPrevError != 0 && prevPrevError - prevError < 0.003)
//					if (epoch % 8 == 0)
//					{
//						System.out.println("Change hyperparameters");
//						Trainer<?> t = (Trainer<?>) event.getSource();
//						Hyperparameters hp = t.getHyperparameters();
//						t.getNeuralNetwork().getConnections().stream().filter(c -> c instanceof WeightsConnections && hp.getLearningRate(c) > 0.00001f).forEach(c -> {
//							hp.setLearningRate(c, hp.getLearningRate(c) / 10);
//						});
//					}
//				}
//			}
//		});

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
		JUnitOnServerStarter jUnitOnServerStarter = new CifarCaffeConf();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
