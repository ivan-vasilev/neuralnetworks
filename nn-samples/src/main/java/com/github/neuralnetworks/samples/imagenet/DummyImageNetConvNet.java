package com.github.neuralnetworks.samples.imagenet;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.CaffeConfigIOUtil;
import com.github.neuralnetworks.input.RandomInputProvider;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Tony
 */
public class DummyImageNetConvNet extends JUnitOnServerStarter
{

	private static final Logger logger = LoggerFactory.getLogger(DummyImageNetConvNet.class);

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

		conf3.getOpenCLConfiguration().setUseOptionsString(true);

		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Override
	public void jUnitTest(int epochs, String folder)
	{

		NeuralNetworkBuilder neuralNetworkBuilder;
		try
		{
			neuralNetworkBuilder = CaffeConfigIOUtil.readBuilderFromClasspath("imagenet/train_val.prototxt", "imagenet/solver.prototxt");
		} catch (IOException e)
		{
			throw new IllegalStateException("Can't load the configuration files!", e);
		}

		// trainer
		{
			int imageSize = 227;

			neuralNetworkBuilder.setTrainingSet(new RandomInputProvider(256 * 2, imageSize * imageSize * 3, 462));
			neuralNetworkBuilder.setTestingSet(new RandomInputProvider(256 * 2, imageSize * imageSize * 3, 462));
		}

//		logger.info(neuralNetworkBuilder.toString());

		// build the network
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = neuralNetworkBuilder.buildWithTrainer();

		BackPropagationTrainer<?> bpt = (BackPropagationTrainer<?>) neuralNetworkTrainerPair.getRight();
		bpt.setEpochs(epochs);


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
		JUnitOnServerStarter jUnitOnServerStarter = new DummyImageNetConvNet();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
