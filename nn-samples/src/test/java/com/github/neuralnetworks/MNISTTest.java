package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputModifier;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;

/**
 * MNIST test
 */
public class MnistTest {

    /**
     * Simple mnist backpropagation test
     */
    @Test
    public void testMnistMultipleSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 784, 2048, 2048, 10 }, true);
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputModifier(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputModifier(255));
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.02f, 0.5f, 0f);
	bpt.addEventListener(new LogTrainingListener(true));

	//Environment.getInstance().setExecutionStrategy(new CPUKernelExecution());

	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
