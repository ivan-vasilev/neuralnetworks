package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.samples.xor.XorInputProvider;
import com.github.neuralnetworks.samples.xor.XorOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.EarlyStoppingListener;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;

public class XorTest {

    /**
     * Simple xor backpropagation test
     */
    @Test
    public void testMLPSigmoidBP() {
	// create multi layer perceptron with one hidden layer and bias
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 8, 1 }, true, true);

	// create training and testing input providers
	XorInputProvider trainingInput = new XorInputProvider();
	XorInputProvider testingInput = new XorInputProvider();

	// create backpropagation trainer for the network
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainingInput, testingInput, new XorOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 1f, 0.5f, 0f, 0f, 1, 1, 100);

	// add logging
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// early stopping
	bpt.addEventListener(new EarlyStoppingListener(testingInput, 10, 0.1f));

	// train
	bpt.train();

	// test
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
