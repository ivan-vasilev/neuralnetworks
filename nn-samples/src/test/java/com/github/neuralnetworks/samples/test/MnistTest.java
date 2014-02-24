package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.util.Environment;

/**
 * MNIST test
 */
public class MnistTest {

    @Ignore
    @Test
    public void testSigmoidBP() {
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 784, 10 }, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Ignore
    @Test
    public void testSigmoidHiddenBP() {
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 784, 300, 100, 10 }, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 2, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Ignore
    @Test
    public void testRBM() {
	RBM rbm = NNFactory.rbm(784, 10, false);
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider,  new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 1, false);

	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    @Ignore
    public void testLeNet() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1 }, { 2, 2 }, { 5, 5, 50, 1 }, { 2, 2 }, {512}, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.01f, 0.5f, 0f, 0f);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	bpt.train();

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testLeNetSmall() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 4, 1 }, { 2, 2 }, { 5, 5, 6, 1 }, { 2, 2 }, {120}, {10} }, false);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.9f, 0f, 0f);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.train();

	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
