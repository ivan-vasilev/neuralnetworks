package com.github.neuralnetworks.samples.test;

import java.io.File;

import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider.CIFAR10TestingInputProvider;
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider.CIFAR10TrainingInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.util.Util;

/**
 * Convert CIFAR and MNIST in more efficient binary arrays
 */
public class CifarMnistToFloatConverter
{
	public static void main(String[] args)
	{
		CIFAR10TrainingInputProvider cifarTrainInputProvider = new CIFAR10TrainingInputProvider("cifar-10-batches-bin-new-mean");
		cifarTrainInputProvider.getProperties().setGroupByChannel(true);
		cifarTrainInputProvider.getProperties().setScaleColors(true);
		cifarTrainInputProvider.getProperties().setSubtractMean(false);
		cifarTrainInputProvider.getProperties().setSubtractArray(Util.readFileIntoFloatArray(new File("C:\\dev\\git_local\\exb\\research_cnn\\ExB NN Samples\\cifar-10-batches-bin-new-mean\\train-mean-values.float")));
		cifarTrainInputProvider.getProperties().setParallelPreprocessing(false);
		cifarTrainInputProvider.reset();
		Util.inputToFloat(cifarTrainInputProvider, "cifar-10-batches-bin-new-mean/train-data.float", "cifar-10-batches-bin-new-mean/train-labels.float");

		CIFAR10TestingInputProvider cifarTestInputProvider = new CIFAR10TestingInputProvider("cifar-10-batches-bin-new-mean");
		cifarTestInputProvider.getProperties().setGroupByChannel(true);
		cifarTestInputProvider.getProperties().setScaleColors(true);
		cifarTestInputProvider.getProperties().setSubtractMean(false);
		cifarTestInputProvider.getProperties().setSubtractArray(Util.readFileIntoFloatArray(new File("C:\\dev\\git_local\\exb\\research_cnn\\ExB NN Samples\\cifar-10-batches-bin-new-mean\\test-mean-values.float")));
		cifarTestInputProvider.getProperties().setParallelPreprocessing(false);
		cifarTestInputProvider.reset();
		Util.inputToFloat(cifarTestInputProvider, "cifar-10-batches-bin-new-mean/test-data.float", "cifar-10-batches-bin-new-mean/test-labels.float");

		MnistInputProvider mnistTrainInputProvider = new MnistInputProvider("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
		mnistTrainInputProvider.addInputModifier(new ScalingInputFunction(255));
		mnistTrainInputProvider.reset();
		Util.inputToFloat(mnistTrainInputProvider, "mnist/train-images.float", "mnist/train-labels.float");

		MnistInputProvider mnistTestInputProvider = new MnistInputProvider("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");
		mnistTestInputProvider.addInputModifier(new ScalingInputFunction(255));
		mnistTestInputProvider.reset();
		Util.inputToFloat(mnistTestInputProvider, "mnist/t10k-images.float", "mnist/t10k-labels.float");
	}
}
