package com.github.neuralnetworks.samples.server;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Continues a training of a cifar or mnist test. Use the default command arguments + "patch to training file" + "type of data set (cifar/mnist) <br>
 * for example: 2 1 networks\network_conv_fc_fc.epoch_1 mnist <br>
 * This class is for debugging, not more!
 * 
 * @author tmey
 */
public class ContinueTrainingOnServer extends JUnitOnServerStarter
{

	private TrainSet trainSet;
	private File trainerFile;

	public ContinueTrainingOnServer(TrainSet trainSet, File trainerFile)
	{
		if (trainSet == null)
		{
			throw new IllegalArgumentException("trainSet must be not null!");
		}

		if (trainerFile == null)
		{
			throw new IllegalArgumentException("trainerFile must be not null!");
		}

		this.trainSet = trainSet;
		this.trainerFile = trainerFile;
	}

	@Override
    public List<RuntimeConfiguration[]> runtimeConfigurations() {
        List<RuntimeConfiguration[]> configurations = new ArrayList<>();

        RuntimeConfiguration conf1 = new RuntimeConfiguration();
        conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
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

		// create input provider
		TrainingInputProvider trainInputProvider;
		TrainingInputProvider testInputProvider;

		switch (trainSet) {
		case Cifar:
			trainInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/train-data.float", "cifar-10-batches-bin/train-labels.float", 32 * 32, 10, 60000);
			testInputProvider = new SimpleFileInputProvider("cifar-10-batches-bin/test-data.float", "cifar-10-batches-bin/test-labels.float", 32 * 32, 10, 10000);
			break;
		case Mnist:
			trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
			testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
			break;
		default:
			throw new IllegalArgumentException("unknown train set: " + trainSet);
		}

		// load trainer
		Trainer<?> trainer;
		try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(trainerFile)))
		{
			trainer = (Trainer<?>) inputStream.readObject();
		} catch (Exception e)
		{
			throw new IllegalStateException("can't load the network from " + trainerFile, e);
		}
		// start training

		trainer.setEpochs(epochs);
		trainer.setTrainingInputProvider(trainInputProvider);
		trainer.setTestingInputProvider(testInputProvider);

		trainer.train();

		// test
		trainer.test();

		// do something with the result
	}

	public enum TrainSet
	{
		Cifar,
		Mnist
	}

	public static void main(String[] args)
	{
		if (args.length < 2)
		{
			throw new IllegalArgumentException("I need minimum two arguments: the save file with the trainer and the type of the training set!");
		}

		String trainsetType = args[args.length - 1];
		TrainSet trainset;
		if (trainsetType.equalsIgnoreCase("cifar"))
		{
			trainset = TrainSet.Cifar;
		} else
		{
			if (trainsetType.equalsIgnoreCase("mnist"))
			{
				trainset = TrainSet.Mnist;
			} else
			{
				throw new IllegalArgumentException("unknown dataset: " + trainsetType);
			}
		}

		String pathToTrainSet = args[args.length - 2];


		String[] shortArgs = new String[args.length - 2];
		System.arraycopy(args, 0, shortArgs, 0, shortArgs.length);

		ContinueTrainingOnServer continueTrainingOnServer = new ContinueTrainingOnServer(trainset, new File(pathToTrainSet));
		continueTrainingOnServer.startTestFromCommandLine(shortArgs);
	}
}
