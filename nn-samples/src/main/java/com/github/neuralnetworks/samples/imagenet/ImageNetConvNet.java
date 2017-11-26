package com.github.neuralnetworks.samples.imagenet;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.CaffeConfigIOUtil;
import com.github.neuralnetworks.input.image.DirectoryImageInputProvider;
import com.github.neuralnetworks.input.image.ImageResizeStrategy;
import com.github.neuralnetworks.samples.server.JUnitOnServerStarter;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Tony
 */
public class ImageNetConvNet extends JUnitOnServerStarter
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

		conf3.getOpenCLConfiguration().setUseOptionsString(true);

		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Override
	public void jUnitTest(int epochs, String folder)
	{

		File trainDirectory = new File("D:\\testDB\\evaluation\\ILSVRC2014_DET_train\\ILSVRC2014_DET_train");
		if (!trainDirectory.exists())
		{
			trainDirectory = new File("/data/pictures/ImageNetChallange/ILSVRC2014_DET_train/");
		}

		if (!trainDirectory.exists())
		{
			trainDirectory = new File("/data/pictures/imagenet/ILSVRC2014_DET_train/");
		}

		File testDirectory = trainDirectory;


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

			DirectoryImageInputProvider trainInputProvider = new DirectoryImageInputProvider(null, trainDirectory);
			trainInputProvider.getProperties().setGroupByChannel(true);
			trainInputProvider.getProperties().setScaleColors(true);
			trainInputProvider.getProperties().setSubtractMean(false);
			trainInputProvider.getProperties().setIsGrayscale(false);
			trainInputProvider.getProperties().setImagesBulkSize(300);
			trainInputProvider.getProperties().setParallelPreprocessing(true);
			trainInputProvider.getProperties().setAugmentedImagesBufferSize(400);
			trainInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, imageSize, false));
			trainInputProvider.reset();

			DirectoryImageInputProvider testInputProvider = new DirectoryImageInputProvider(null, testDirectory);
			testInputProvider.getProperties().setGroupByChannel(true);
			testInputProvider.getProperties().setScaleColors(true);
			testInputProvider.getProperties().setSubtractMean(false);
			testInputProvider.getProperties().setIsGrayscale(false);
			testInputProvider.getProperties().setImagesBulkSize(300);
			testInputProvider.getProperties().setParallelPreprocessing(true);
			testInputProvider.getProperties().setAugmentedImagesBufferSize(400);
			testInputProvider.getProperties().setResizeStrategy(new ImageResizeStrategy(ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE, imageSize, false));
			testInputProvider.reset();

			neuralNetworkBuilder.setTrainingSet(trainInputProvider);
			neuralNetworkBuilder.setTestingSet(testInputProvider);
		}

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
		JUnitOnServerStarter jUnitOnServerStarter = new ImageNetConvNet();
		jUnitOnServerStarter.startTestFromCommandLine(args);

	}
}
