package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.ScalingInputModifier;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistMultipleNeuronsOutputError;
import com.github.neuralnetworks.samples.mnist.MnistSingleNeuronOutputError;
import com.github.neuralnetworks.samples.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.samples.mnist.MnistTargetSingleNeuronOutputConverter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.AparapiXORShiftInitializer;

/**
 * MNIST test
 */
public class MnistTest {

    /**
     * Simple mnist backpropagation test
     */
    @Test
    public void testMNISTSingleSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 784, 500, 10 }, true);
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputModifier(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputModifier(255));
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MnistMultipleNeuronsOutputError(), new AparapiXORShiftInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
    
    /**
     * Simple mnist backpropagation test
     */
    @Ignore
    @Test
    public void testMNISTMultiSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 784, 500, 1 }, true);
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000, new MnistTargetSingleNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputModifier(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, new MnistTargetSingleNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputModifier(255));
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MnistSingleNeuronOutputError(), new AparapiXORShiftInitializer(-0.01f, 0.01f), 0.1f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

}
