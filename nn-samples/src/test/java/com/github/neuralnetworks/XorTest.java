package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.samples.xor.XorInputProvider;
import com.github.neuralnetworks.samples.xor.XorOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;

public class XorTest {

    /**
     * Simple xor backpropagation test
     */
    @Test
    public void testMLPSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 2, 8, 1 }, true);
	XorInputProvider trainingInput = new XorInputProvider(10000);
	XorInputProvider testingInput = new XorInputProvider(4);
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainingInput, testingInput, new XorOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 1f, 0.5f, 0f);
	bpt.addEventListener(new LogTrainingListener());
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
