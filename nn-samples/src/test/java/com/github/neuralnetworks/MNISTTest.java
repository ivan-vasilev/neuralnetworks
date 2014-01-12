package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.samples.mnist.MnistInputConverter;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistMultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.AparapiXORShiftInitializer;

/**
 * MNIST test
 */
public class MNISTTest {

    /**
     * Simple mnist backpropagation test
     */
    @Test
    public void testMNISTSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 784, 500, 10 }, false);
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-labels.idx1-ubyte", "train-images.idx3-ubyte", 10, new MnistInputConverter(), new MnistInputConverter());
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", 10, new MnistInputConverter(), new MnistInputConverter());
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MnistMultipleNeuronsOutputError(), new AparapiXORShiftInitializer(-0.01f, 0.01f), 0.1f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

}
