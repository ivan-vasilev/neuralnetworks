package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.samples.iris.IrisInputProvider;
import com.github.neuralnetworks.samples.iris.IrisTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;

/**
 * Iris test
 */
public class IrisTest {

    /**
     * Simple iris backpropagation test
     */
    @Test
    public void testIrisMultipleSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 4, 8, 3 }, true);
	IrisInputProvider trainInputProvider = new IrisInputProvider(1, 1, new IrisTargetMultiNeuronOutputConverter(), true, true);
	IrisInputProvider testInputProvider = new IrisInputProvider(150, 150, new IrisTargetMultiNeuronOutputConverter(), false, true);
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.2f, 0.2f), 0.1f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);

	bpt.addEventListener(new LogTrainingListener());

	bpt.train();
	LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(mlp.getOutputLayer());
	//cc.addActivationFunction(new SoftmaxFunction());

	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * Simple iris backpropagation test
     */
    @Ignore
    @Test
    public void testIrisMultipleSoftReLUBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSoftRelu(new int[] { 4, 12, 12, 3 }, true, new AparapiSigmoid());
	IrisInputProvider trainInputProvider = new IrisInputProvider(1000, 1000, new IrisTargetMultiNeuronOutputConverter(), true, true);
	IrisInputProvider testInputProvider = new IrisInputProvider(1000, 1000, new IrisTargetMultiNeuronOutputConverter(), false, true);
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 1f, 0.5f, 0f);
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();

	LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(mlp.getOutputLayer());
	cc.addActivationFunction(new SoftmaxFunction());

	bpt.test();
	System.out.println("Error: " + bpt.getOutputError().getTotalNetworkError());
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
