package com.github.neuralnetworks.samples.server.cifar;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.EpochFinishedEvent;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.events.ValidationListener;
import com.github.neuralnetworks.training.random.GaussianRandomInitializerImpl;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * @author Urko
 * This class contains the model that produces better results so far (28/01/2015). DO NOT MODIFY! 
 * The accuracy can be checked here https://leipzig.exb.de/jira/browse/VISION-534
 */
public class BaselineModel extends JUnitOnServerStarter {

	@Override
	public List<RuntimeConfiguration[]> runtimeConfigurations() {
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
		conf4.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf4.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf4.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf4.getOpenCLConfiguration().setUseOptionsString(false);
		conf4.getOpenCLConfiguration().setRestartLibraryAfterPhase(true);
		conf4.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf4 });

		// Old configuration
		RuntimeConfiguration conf = new RuntimeConfiguration();
		conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		conf.setUseDataSharedMemory(false);
		conf.setUseWeightsSharedMemory(false);
		conf.getOpenCLConfiguration().setAggregateOperations(true);
		conf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf });

		return configurations;
	}

	@Override
	public void jUnitTest(int epochs, String folder) {
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		Random r = new Random();
		long seed = 123456;

		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32,	3));

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (3 * 5 * 5 + 32 * 5 * 5))));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 32);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (32 * 5 * 5 + 32 * 5 * 5))));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasMomentum(0.9f);
				convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(3);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setActivationType(ActivationType.Nothing);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// conv
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 64);
				convolutionalLayerBuilder.setPaddingSize(0);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (32 * 5 * 5 + 64 * 5 * 5))));
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
				fullyConnectedLayerBuilder.setWeightInitializer(new GaussianRandomInitializerImpl(r, seed, 0, 0.01f, 1 / Math.sqrt(6 * (64 * 5 * 5 + 10))));
				fullyConnectedLayerBuilder.setLearningRate(0.05f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.05f);
				fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}

			// trainer
			{
				TrainingInputProvider trainInputProvider = new SimpleFileInputProvider(folder + "/train-data.float", folder + "/train-labels.float", 32 * 32, 10, 50000);
				TrainingInputProvider testInputProvider = new SimpleFileInputProvider(folder + "/test-data.float", folder + "/test-labels.float", 32 * 32, 10, 10000);
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

		//Train error per epoch. Change the dataset in case you need another kind of error
		bpt.addEventListener(new ValidationListener(new SimpleFileInputProvider(folder + "/train-data.float", folder + "/train-labels.float", 32 * 32, 10, 50000)) {
			private static final long serialVersionUID = 1L;

			private int epoch;

			@Override
			public void handleEvent(TrainingEvent event) {
				super.handleEvent(event);
				if (event instanceof EpochFinishedEvent) {
					epoch++;
					if (epoch % 8 == 0) {
						System.out.println("Change hyperparameters");
						Trainer<?> t = (Trainer<?>) event.getSource();
						Hyperparameters hp = t.getHyperparameters();
						t.getNeuralNetwork().getConnections().stream().filter(c -> c instanceof WeightsConnections && hp.getLearningRate(c) > 0.00001f)
								.forEach(c -> {hp.setLearningRate(c,hp.getLearningRate(c) / 10);});
					}
				}
			}
		});

		// training
		bpt.train();

		// testing
		bpt.test();

		if (bpt.getOutputError().getTotalNetworkError() > 0.1) {
			throw new IllegalStateException("error was to high! "
					+ bpt.getOutputError().getTotalNetworkError() + " > 0.1");
		}

	}

	public static void main(String[] args) {
		JUnitOnServerStarter jUnitOnServerStarter = new BaselineModel();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}

}
