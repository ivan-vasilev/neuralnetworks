package com.github.neuralnetworks.samples.test;

import java.util.Random;

import org.junit.Test;

import com.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider.CIFAR10TestingInputProvider;
import com.github.neuralnetworks.samples.cifar.CIFARInputProvider.CIFAR10TrainingInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;

/**
 * CIFAR test
 */
public class CifarTest {

    /**
     * NOT TESTED
     */
    @Test
    public void testSigmoidBP() {
	Environment.getInstance().setUseDataSharedMemory(false);
	Environment.getInstance().setUseWeightsSharedMemory(false);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 3072, 10 }, true);

	CIFAR10TrainingInputProvider trainInputProvider = new CIFAR10TrainingInputProvider("cifar-10-batches-bin"); // specify your own path
	trainInputProvider.getProperties().setGroupByChannel(true);
	trainInputProvider.getProperties().setScaleColors(true);
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));

	// specify your own path
	CIFAR10TestingInputProvider testInputProvider = new CIFAR10TestingInputProvider("cifar-10-batches-bin"); // specify your own path
	testInputProvider.getProperties().setGroupByChannel(true);
	testInputProvider.getProperties().setScaleColors(true);
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new RandomInitializerImpl(new Random(), -0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 0f, 1, 1000, 1);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.train();
	bpt.test();
    }
}
