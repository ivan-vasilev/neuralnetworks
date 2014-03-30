package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.Trainer;
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

	public static final String RESOURCES_PATH = "\\java\\resources\\ML\\datasets\\mnist\\";
	private static final String T10K_LABELS_IDX1_UBYTE = RESOURCES_PATH+"t10k-labels.idx1-ubyte";
	private static final String T10K_IMAGES_IDX3_UBYTE = RESOURCES_PATH+"t10k-images.idx3-ubyte";
	private static final String TRAIN_LABELS_IDX1_UBYTE =RESOURCES_PATH+ "train-labels.idx1-ubyte";
	private static final String TRAIN_IMAGES_IDX3_UBYTE =RESOURCES_PATH+ "train-images.idx3-ubyte";
	
	
    @Test
    public void testSigmoidBP() {
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 784, 10 }, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1000, 1, new MnistTargetMultiNeuronOutputConverter());
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

	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 2, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testRBM() {
	RBM rbm = NNFactory.rbm(784, 10, false);
	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider,  new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 1, false);

	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }


    @Test
    public void testAE() {
	Autoencoder nn = NNFactory.autoencoderSigmoid(784, 10, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	Trainer<?> t = TrainerFactory.backPropagationAutoencoder(nn, trainInputProvider, testInputProvider,  new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 0f);

	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	t.train();
	nn.removeLayer(nn.getOutputLayer());
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testLeNetSmall() {
	// Convolutional network
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1 }, { 2, 2 }, { 5, 5, 50, 1 }, { 2, 2 }, {512}, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// Mnist dataset provider
	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1000, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.01f, 0.5f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	// cpu execution mode
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testLeNetTiny() {
	// very simple convolutional network with a single 2x2 max pooling layer
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 2, 2 }, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// MNIST dataset
	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	// cpu execution
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }


    /**
     * MNIST small LeNet network
     */
    @Test
    public void testLeNetTiny2() {
	// very simple convolutional network with a single convolutional layer with 6 5x5 filters and a single 2x2 max pooling layer
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 6, 1 }, {2, 2}, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// MNIST dataset
	MnistInputProvider trainInputProvider = new MnistInputProvider(TRAIN_IMAGES_IDX3_UBYTE, TRAIN_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider(T10K_IMAGES_IDX3_UBYTE, T10K_LABELS_IDX1_UBYTE, 1, 1, new MnistTargetMultiNeuronOutputConverter());
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	// cpu execution
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
