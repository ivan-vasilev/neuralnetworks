package com.github.neuralnetworks.analyzing.network.comparison;

import java.io.File;

import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Serializer;

/**
 * @author tmey
 */
public class CompareSavedNetworks
{

	public static void compareNetworks(File trainerOneFile, File trainerTwoFile, boolean compareOnlyWeights) throws DifferentNetworksException
	{
		Trainer<?> trainerOne = Serializer.loadTrainer(trainerOneFile.getAbsolutePath());
		Trainer<?> trainerTwo = Serializer.loadTrainer(trainerTwoFile.getAbsolutePath());

		NetworkActivationAndWeightComparison networkActivationAndWeightComparison = new NetworkActivationAndWeightComparison();
		networkActivationAndWeightComparison.setCompareActivation(!compareOnlyWeights);

		networkActivationAndWeightComparison.compareTrainedNetworks((BackPropagationTrainer<?>) trainerOne, (BackPropagationTrainer<?>) trainerTwo);
	}


	public static void main(String[] args) throws DifferentNetworksException
	{

		if (args.length != 3)
		{
			throw new IllegalArgumentException("There must be three arguments: " +
					"first trainer file, second trainer file and a boolean flag if only the weights should be compares! (in that order!)");
		}

		CompareSavedNetworks.compareNetworks(new File(args[0]), new File(args[1]), Boolean.parseBoolean(args[2]));
	}
}
