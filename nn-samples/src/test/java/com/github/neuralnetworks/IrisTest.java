package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.samples.iris.IrisInputProvider;
import com.github.neuralnetworks.samples.iris.IrisTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;

/**
 * Iris test
 */
public class IrisTest {

    /**
     * Simple mnist backpropagation test
     */
    @Test
    public void testIrisMultipleSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 4, 12, 12, 3 }, true);
	IrisInputProvider trainInputProvider = new IrisInputProvider(1000, 100000, new IrisTargetMultiNeuronOutputConverter());
	IrisInputProvider testInputProvider = new IrisInputProvider(1000, 10000, new IrisTargetMultiNeuronOutputConverter());
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.02f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
